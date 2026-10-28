import json, glob, sys, os
from collections import Counter
# 用法: python3 fsdp_nan_class.py <RUN目录>  (RUN = config dump_path 的父目录,其下有 dump_actor/)
base = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('RUN_DIR', '')
if not base or not os.path.isdir(base + '/dump_actor'):
    raise SystemExit("用法: python3 fsdp_nan_class.py <RUN目录>（其下须有 dump_actor/；容器内跑,dump.json 是 root 0750）")
af = sorted(glob.glob(base + '/dump_actor/step*/rank0/dump.json'))
print('actor dump.json (rank0):', len(af))
if not af:
    raise SystemExit('no actor dump')
f = af[len(af) // 2]
print('inspect:', os.path.relpath(f, base))
d = json.load(open(f)).get('data', {})
print('total modules:', len(d))

def cls(name):
    n = name.lower()
    for t in ['cross_entropy', 'crossentropy', 'lm_head', 'o_proj', 'q_proj', 'k_proj', 'v_proj',
              'down_proj', 'gate_proj', 'up_proj', 'input_layernorm', 'post_attention', 'rmsnorm',
              'attn', 'silu', 'act_fn', 'embed', 'rotary']:
        if t in n:
            return t
    return 'other'

# 区分:NaN 在 input 还是 output[0](真值)还是其它 output buffer(可能良性)
nan_in_output0 = []   # output[0] 含 NaN = 真信号
nan_other = []        # 仅 input / 非 output[0] buffer 含 NaN
for k, v in d.items():
    outs = v.get('output', []) if isinstance(v, dict) else []
    has_nan_out0 = False
    if outs and isinstance(outs[0], dict):
        s0 = json.dumps(outs[0])
        if 'NaN' in s0:
            has_nan_out0 = True
    whole = json.dumps(v)
    if 'NaN' not in whole:
        continue
    if has_nan_out0:
        nan_in_output0.append(k)
    else:
        nan_other.append(k)

print('=== 含 NaN 的模块拆解 ===')
print('output[0] 含 NaN(真信号,需排查):', len(nan_in_output0))
print('   类型分布:', dict(Counter(cls(m) for m in nan_in_output0)))
for m in nan_in_output0[:10]:
    print('     ', m[:95])
print('仅 input/其它 buffer 含 NaN(可能良性 CE 未初始化缓冲):', len(nan_other))
print('   类型分布:', dict(Counter(cls(m) for m in nan_other)))
for m in nan_other[:6]:
    print('     ', m[:95])
