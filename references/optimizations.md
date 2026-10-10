# �����Ż���Ǩ�Ʒ�Χ��·����֤��

��ѡ���û��漰���У��ٶ���� skill��������ʱ�ɼ����Ѱ�װ skill �嵥��λ����ȡ��Ӧ `SKILL.md`��������������·��д���Ǩ����Ŀ��������ɵ��������ᵽ�����й��ߡ�

| �Ż����� | Ǩ��ʱ�˶���Ķ� | �ɸ��õ����� skill | �����ص� |
|---|---|---|---|
| �ں����ӿ��� | ��ͼ���塢���������ߡ�shape/dtype/layout��ע��/�󶨡�tiling����ֵ��ͬ�� | `model-infer-fusion`�����Ϸ���ʱ `ascendc-cross-gen-port` / `ascendc-cross-gen-port-light`����ʵ�������� AscendC/Triton/TileLang ��������֤ skill | ʵ�����С����Ӿ��ȡ�����ģ�͡�Ŀ��ؼ�·���� E2E |
| ��������Ż� | runner/model �ӿڡ����ע�ᡢȨ��/metadata��ͼ����/replay�����뻺�桢������� | `model-infer-migrator`������Ŀ�겿����ߣ���`model-infer-graph-mode`��`torch-custom-ops-guide` | ͬ��ܰ汾�� eager/ͼ·�����ӿڼ��ݡ���̬����������� |
| Prefill �׶��Ż� | packed/�䳤���롢mask��position��chunk �߽硢KV ����д�롢�ں�/ͨ�Ų��� | `model-infer-fusion`��`model-infer-kvcache`��`model-infer-profiling` | Prefill ���ȡ�TTFT/�׶κ�ʱ���������¡������������� Decode ���� |
| Decode �׶��Ż� | decode batch��KV ����/��ַ�仯��ͼ bucket��ѭ��״̬��Ȩ��Ԥȡ��ͨ������ | `model-infer-graph-mode`��`model-infer-prefetch`��`model-infer-multi-stream` | �����ಽ���ȡ�TPOT/ITL��p95��������¡�capture/replay |
| �ڴ��Ż� | weight/KV/workspace/graph pool ��ֵ���������ͷš�alias������/Ԥȡ/offload �������� | `model-infer-kvcache`��`model-infer-prefetch`��GE ������ `ge-memory-analysis` | ʵ�ʷ�ֵ����̬�ڴ桢�����˳���Ļ��ա�OOM �߽缰���˿��� |
| KV Cache �Ż� | ����/��ҳ/MLA/��ϲ�/������block/slot������ scale/offset��prefix ���á�ͼ��ַ | `model-infer-kvcache`��`model-infer-quantization`��`model-infer-precision-debug` | д���ȼۡ������к� block �߽硢ҳ����/���ա����������Ⱥ� TPOT |
| ���в����Ż� | TP/EP/DP/PP/CP ��ʵ�ʿ���/���ˡ�Ȩ�� shard��heads/expert ������collective ˳�� | `model-infer-parallel-analysis`��`model-infer-parallel-impl`��`ascendc-mc2-best-practice` | ȫ ranks ���/�˳���ͨ���������š�ģ�;��ȡ�����/�ӳ�/�ڴ� |
| ���Ȳ����Ż� | continuous batching��chunk��׼��/���ȼ���token budget��preemption������״̬��KV ��� | Ŀ�������� scheduler���ӿ��ĵ��Ͳ��ԣ��� `model-infer-profiling` �������� | ��ϳ��������Ŷ�/p95��SLO �����¡���ƽ�ԡ�ȡ��/��ʱ/��ռ�����Դ���� |
| PD ���볡���Ż� | P/D ��ɫ��KV ����Э���벼�֡���/shard ӳ�䡢Ԫ���ݡ�connector��������·�� | Ŀ�������� connector/transfer ʵ�ֺͼ��ɲ��ԣ�KV/���� skills ��������ʵ�ʸ��������� | ���ػ��ߵ�������·�ľ��ȡ�����ʱ�ӡ�TTFT/TPOT��SLO ���¡�ʧ�ܺ���� |

## ·�ɱ߽�

- �����˶Կ�ʹ�� `ascendc-env-check`������Զ��ִ�й��������� `remote-cann-development`������Ϊ��Ǩ��ǿ������һ��Զ�̺�ˡ�
- ���ܲɼ��������п�� profiler����Ҫʱ�� `model-infer-profiling`��`ops-profiling`�������� `model-infer-perf-breakdown`��ԭʼ trace �����׶Ρ�rank/device��shape ������߽硣
- �Ѱ�װ�Ŀ������ skill �з�����������ƣ���Ӳ�����ϡ����ⷽ����ɷ�����������ר������֧�����ⷽ��
- `model-infer-*` �в���Լ����� cann-recipes-infer ����/BF16 ���ض�����ģʽ������ vLLM-Ascend ���������ʱ��ֻ�������ü���ָ�����˶���ʵ�ӿڣ����հ�Ŀ¼��Ĭ�ϻ���ģʽ�������ֶΡ�
- `model-infer-graph-mode` �Ľ׶�Լ�����ܴ���Ŀ����֧�־���Prefill/Decode �ܷ���ͼȡ����Ŀ����ʵ·������֤��
- δ�ṩ��������Ǩ�ƻ� PD ȫ���� skill ʱ��ֱ�ӻ���Ŀ��ʵ��������䣻�����鹹 skill ����ѵ������з���������� PD ��ʵ�����ݡ�
- ȱ�ٿ�ѡ skill ʱʹ���Ѻ�ʵ��Ŀ�깤��/�ٷ��ĵ�������ֻ�б�Ҫ��Դȱʧ���޿������ʱ����������������

## �����Ż������

��ʵ������ʵʩ��Ȩ��/���в��ֱ仯����Ӱ���ں����롢KV �ṹ�� PD ���䣻������ͼ��ַԼ������Ӱ����Ⱥ���ռ�����ȵ� batch/chunk �仯���ܸı� Prefill/Decode ·����

����������������������֤����¼�����������ϱȵ����˻�ʱ������������ͨ���ص���capture bucket��workspace ��ֵ�͵��ȵȴ����û���Ҫ��Ǩ��ָ���Ż�ʱ�������������µ����������л���Ȳ��ԡ�

PD �� P/D ģ��Ȩ�ء�tokenizer �������������һ�£���ģ��Ǩ���ǰѷ���Ǩ���µ�ģ�ͷ��񣬲��ǰ�ģ�ͼײ����� KV ֱ�ӹ�ģ���ҽ��롣Ŀ��� P/D Ӳ���Ƿ�ͬ���������Ƿ�ɿ�Ӳ�������û����𼰺��֤��Ϊ׼��
