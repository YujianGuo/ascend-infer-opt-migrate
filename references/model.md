# ģ���᣺Ǩ�Ƽ���������������Լ

���� M �� HM����Ŀ��ģ�ͱ����Ĳο�ʵ������֤����Ҫ��Դģ�ͺ�Ŀ��ģ��������ͬ�ı���

## �ṹӳ��

��ģ�� config��forward��Ȩ�ؼ��غ�ʵ�� tensor ȡ���±���Ϣ����Ҫֻ����ģ������module ����һ�žɽṹͼ��

| ģ����Լ | �������� | ����Ǩ�Ʒ��� |
|---|---|---|
| Attention | MHA/GQA/MQA/MLA������/ϡ�衢��� full/linear attention������� | ��ѹ�� latent KV ������ K/V����������״̬�� attention cache |
| ά�� | hidden/head_dim��Q/KV heads��������shard ��״��packed/�䳤���� | TP heads �зֲ��������̶�ά�� kernel Խ�� |
| RoPE/Norm | ��ͨ RoPE/MRoPE����ת��Χ/˳��λ�����롢scale��epsilon��RMS weight offset���в����� | �����״��ͬ����ѧ���岻ͬ |
| FFN/MoE | ���gate/up/down ˳��ר����/top-k��·��Ȩ�ء�����ר�� | �ſ�/ר��˳��·�ɺ� combine ����ı� |
| Ȩ��/���� | ����ӳ�䡢QKV packing��ת�á�weight loader��post-load��K/V scale �� offset | Ԫ������ stacked ���������кϲ���ʧ |
| ���� | cache ���͡�ÿ��ṹ��layout/block��slot��position������Ԫ���ݡ�״̬���� | ���滻��ȡ���ӣ���©д��/����/ͼ replay |
| ���нӿ� | forward metadata����������ߡ�hook�����֧��in-place/alias��ͼģʽ | �ں��Ƴ��Ա��ⲿ���ѵ��м�ֵ |
| ��ģ̬ | vision/���Ա߽硢processor���Ӿ� token �� position���䳤ƴ�� | ��������ģ�� shape ��λ�ü��� |

## �ɸ������ж�

��ÿ��Դ�Ż�����Ŀ����ͼ����������������ʱ���ã���ͬ���ɽ����ȼ�ӳ��ʱ���䣻û�ж�Ӧ�ṹʱ��ǲ����á����� MoE dispatch ����Ǩ��û��ר�ҽṹ�� Dense FFN��MLA ѹ�����治�ܽ������ֳ�Ϊ GQA cache��

�̶�ά���ں�Ǩ��ǰ���Ŀ�� head_dim��stride�������ԡ�device��dtype��β��Ϳ�����֧�֣�Ŀ�겻����ʱ���ڱ�Ҫ�߽�ת���������Ϸ����ˡ��ںϷ�Χ������������Ϊ�磬���ܶ��� hook/graph/debug ����Ҫ��ֵ��

������ KV Ǩ�Ʒֱ�׷�� K/V �� scale��offset/zero-point �� checkpoint �� loader��cache д�롢Attention ���ѡ�ͼ����� replay ����·����� attention ģ��ֻ�����ò����ö�Ӧ�Ż������� full-attention KV �Ż���չ������/ѭ��״̬��

## ���ߺͲ���

Ŀ�� baseline �̶�Ŀ��Ȩ�ء�tokenizer/processor���������á�������ο��㷨��candidate �����Ӵ�����Ǩ�ơ��漰Ȩ�ر任ʱ�˶� key ���ǡ���״���з�/�ϲ��ͺ���������֤�任ǰ��ȼ��ԡ�

����ʹ��Ŀ������ goldens�����Ӳο������⼯��û�п����вο�ʱ����׼�������벹������ģ�;��ȱ���δ��֤��������ɶ�������ģ�;���ͨ����

С��ģ�ȶ�λ�״η��磺ģ�������Prefill logits���׸� Decode step��������� Decode step��cache/״̬���£��ٰ�����Ҫ����ɶ˵������⡣�Ƚ�Ŀ��ͬģ�� baseline/candidate ����ֵ����ȷ�������������/��ȷ����·��ʹ����Ŀ���б�׼������ʱ�ſ���ֵ��

������֤�䳤 batch���������ġ�block �߽硢�����ضϡ���ϲ㡢��ģ̬ position ����Ӱ������������������Լ�����в��Թ淶�����������޹���� shape ��������ؼ��߽縲�ǡ�
