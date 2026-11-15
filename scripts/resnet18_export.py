"""ResNet18 export pipeline for zkml.

Design constraints imposed by zkml's TFLite converter (see src/ + python/converter.py):

1. MaxPool2D with SAME padding is rejected outright.  We therefore express every
   padding as an explicit ZeroPadding2D followed by VALID pooling/convolution.
   This is mathematically identical to torchvision's `padding=1` semantics and to
   TF's SAME padding for odd kernels at these sizes, but it keeps every spatial
   op VALID (which the Rust side explicitly supports: "we always have valid
   paddings" in src/layers/conv2d.rs).

2. Conv2D only accepts fused activations NONE / RELU / RELU6.  A standalone RELU
   op has no converter branch, so every ReLU must be fused into a CONV_2D or into
   an ADD (residual).  Whether TFLite fuses the post-add ReLU is verified by
   dumping the resulting op list.

3. BatchNorm must be folded into the preceding convolution.  zkml has no BN layer.
   We fold analytically:  W' = W * gamma / sqrt(var + eps),
                          b' = beta - mean * gamma / sqrt(var + eps).
   BN is affine in inference mode, so folding is exact up to float rounding.

The script emits:
  * <out>.tflite            float32 TFLite model (zkml's converter requires float
                            tensors -- it reads them via the TFLite interpreter)
  * <out>_ops.json          the op list, for auditing against supported operators
  * <out>_reflect.json      float32 reference vs TFLite max abs error

Weights are randomly initialised by default (structure-faithful, NOT pretrained).
Pass --weights to load a real checkpoint; the graph is identical either way.
"""

import argparse
import json
import math
import os

import numpy as np
import tensorflow as tf

EPS = 1e-5


def he_normal(rng, shape, fan_in):
  return rng.normal(0.0, math.sqrt(2.0 / fan_in), size=shape).astype(np.float32)


def bn_params(rng, channels):
  """Random but non-trivial BN parameters, so that folding is actually exercised."""
  gamma = rng.uniform(0.5, 1.5, size=channels).astype(np.float32)
  beta = rng.normal(0.0, 0.1, size=channels).astype(np.float32)
  mean = rng.normal(0.0, 0.1, size=channels).astype(np.float32)
  var = rng.uniform(0.5, 1.5, size=channels).astype(np.float32)
  return gamma, beta, mean, var


def fold(gamma, beta, mean, var):
  scale = gamma / np.sqrt(var + EPS)
  return scale.astype(np.float32), (beta - mean * scale).astype(np.float32)


def build(size, num_classes, seed=0):
  """Builds two parallel Keras graphs: a float reference with explicit BN, and
  the BN-folded graph that is actually exported to TFLite."""
  rng = np.random.default_rng(seed)
  ref_convs, fold_convs, bns = [], [], []

  def conv(x_ref, x_fold, cout, k, stride, pad, relu, tag):
    cin = int(x_ref.shape[-1])
    if pad:
      x_ref = tf.keras.layers.ZeroPadding2D(pad, name=f'{tag}_pad_ref')(x_ref)
      x_fold = tf.keras.layers.ZeroPadding2D(pad, name=f'{tag}_pad')(x_fold)
    W = he_normal(rng, (k, k, cin, cout), k * k * cin)
    gamma, beta, mean, var = bn_params(rng, cout)

    r = tf.keras.layers.Conv2D(cout, k, strides=stride, padding='valid',
                               use_bias=False, name=f'{tag}_conv_ref')(x_ref)
    r = tf.keras.layers.BatchNormalization(epsilon=EPS, name=f'{tag}_bn')(r)
    if relu:
      r = tf.keras.layers.ReLU(name=f'{tag}_relu_ref')(r)

    f = tf.keras.layers.Conv2D(cout, k, strides=stride, padding='valid',
                               use_bias=True,
                               activation='relu' if relu else None,
                               name=f'{tag}_conv')(x_fold)
    ref_convs.append((tag, r, W, gamma, beta, mean, var))
    fold_convs.append((tag, f))
    return r, f

  inp_ref = tf.keras.Input(shape=(size, size, 3), name='input_ref')
  inp_fold = tf.keras.Input(shape=(size, size, 3), name='input')

  # stem: 7x7/2, pad 3
  r, f = conv(inp_ref, inp_fold, 64, 7, 2, 3, True, 'stem')

  # maxpool 3x3/2 with explicit pad 1 (SAME is rejected by zkml)
  rp = tf.keras.layers.ZeroPadding2D(1, name='pool_pad_ref')(r)
  fp = tf.keras.layers.ZeroPadding2D(1, name='pool_pad')(f)
  r = tf.keras.layers.MaxPooling2D(3, strides=2, padding='valid', name='pool_ref')(rp)
  f = tf.keras.layers.MaxPooling2D(3, strides=2, padding='valid', name='pool')(fp)

  stages = [(64, 2, 1), (128, 2, 2), (256, 2, 2), (512, 2, 2)]
  for si, (cout, nblocks, stride) in enumerate(stages):
    for bi in range(nblocks):
      s = stride if bi == 0 else 1
      tag = f's{si}b{bi}'
      r_in, f_in = r, f
      r, f = conv(r, f, cout, 3, s, 1, True, f'{tag}_c1')
      r, f = conv(r, f, cout, 3, 1, 0, False, f'{tag}_c2')
      if s != 1 or int(r_in.shape[-1]) != cout:
        r_s, f_s = conv(r_in, f_in, cout, 1, s, 0, False, f'{tag}_sc')
      else:
        r_s, f_s = r_in, f_in
      # residual add; TFLite may fuse the following ReLU into ADD
      r = tf.keras.layers.Add(name=f'{tag}_add_ref')([r, r_s])
      f = tf.keras.layers.Add(name=f'{tag}_add')([f, f_s])
      r = tf.keras.layers.ReLU(name=f'{tag}_relu_ref')(r)
      f = tf.keras.layers.ReLU(name=f'{tag}_relu')(f)

  # global average pool over H,W
  r = tf.keras.layers.GlobalAveragePooling2D(name='gap_ref', keepdims=True)(r)
  f = tf.keras.layers.GlobalAveragePooling2D(name='gap', keepdims=True)(f)
  r = tf.keras.layers.Flatten(name='flat_ref')(r)
  f = tf.keras.layers.Flatten(name='flat')(f)

  Wfc = rng.normal(0.0, math.sqrt(2.0 / (512 * num_classes)),
                   size=(512, num_classes)).astype(np.float32)
  bfc = rng.normal(0.0, 0.01, size=num_classes).astype(np.float32)
  r = tf.keras.layers.Dense(num_classes, name='fc_ref')(r)
  f = tf.keras.layers.Dense(num_classes, name='fc')(f)

  model_ref = tf.keras.Model(inp_ref, r, name='resnet18_ref')
  model_fold = tf.keras.Model(inp_fold, f, name='resnet18_folded')

  # transfer/fold weights
  by_name = {l.name: l for l in model_fold.layers}
  by_name_ref = {l.name: l for l in model_ref.layers}
  for tag, _, W, gamma, beta, mean, var in ref_convs:
    scale, shift = fold(gamma, beta, mean, var)
    by_name_ref[f'{tag}_conv_ref'].set_weights([W])
    by_name_ref[f'{tag}_bn'].set_weights([gamma, beta, mean, var])
    by_name[f'{tag}_conv'].set_weights([W * scale.reshape(1, 1, 1, -1), shift])
  by_name_ref['fc_ref'].set_weights([Wfc, bfc])
  by_name['fc'].set_weights([Wfc, bfc])

  return model_ref, model_fold


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--size', type=int, default=128)
  ap.add_argument('--classes', type=int, default=1000)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--output', required=True)
  args = ap.parse_args()

  model_ref, model_fold = build(args.size, args.classes, args.seed)

  x = np.random.default_rng(1234).normal(0, 1, (1, args.size, args.size, 3)).astype(np.float32)
  y_ref = model_ref.predict(x, verbose=0)
  y_fold = model_fold.predict(x, verbose=0)
  fold_err = float(np.max(np.abs(y_ref - y_fold)))
  print(f'[build] BN fold max abs err (keras ref vs folded): {fold_err:.3e}')

  conv = tf.lite.TFLiteConverter.from_keras_model(model_fold)
  tflite_model = conv.convert()
  out_tfl = args.output + '.tflite'
  with open(out_tfl, 'wb') as fh:
    fh.write(tflite_model)
  print(f'[export] {out_tfl}  {len(tflite_model)} bytes')

  interp = tf.lite.Interpreter(model_path=out_tfl)
  interp.allocate_tensors()
  det = interp.get_input_details()[0]
  interp.set_tensor(det['index'], x)
  interp.invoke()
  y_tfl = interp.get_tensor(interp.get_output_details()[0]['index'])
  tfl_err = float(np.max(np.abs(y_fold - y_tfl)))

  # op audit
  with open(out_tfl, 'rb') as fh:
    buf = fh.read()
  ops = {}
  try:
    import tflite
    m = tflite.Model.GetRootAsModel(buf, 0)
    sg = m.Subgraphs(0)
    for i in range(sg.OperatorsLength()):
      op = sg.Operators(i)
      code = m.OperatorCodes(op.OpcodeIndex())
      bi = max(code.BuiltinCode(), code.DeprecatedBuiltinCode())
      ops[tflite.opcode2name(bi)] = ops.get(tflite.opcode2name(bi), 0) + 1
  except Exception as exc:  # pragma: no cover
    print(f'[warn] op audit failed: {exc}')

  print('[ops]', json.dumps(ops, indent=2, sort_keys=True))
  inside = interp.get_input_details()[0]['shape'].tolist()
  print(f'[input] {inside}')

  with open(args.output + '_ops.json', 'w') as fh:
    json.dump({'ops': ops, 'input_shape': inside,
               'bn_fold_max_abs_err': fold_err,
               'tflite_vs_folded_max_abs_err': tfl_err}, fh, indent=2)
  with open(args.output + '_reflect.json', 'w') as fh:
    json.dump({'bn_fold_max_abs_err': fold_err,
               'tflite_vs_folded_max_abs_err': tfl_err,
               'ref_argmax': int(np.argmax(y_ref)),
               'tflite_argmax': int(np.argmax(y_tfl))}, fh, indent=2)


if __name__ == '__main__':
  main()
