1.�ο�1_FE_CNT_3.ipynb����������������ΪC_(N)����ͳ��ÿ��������ȡֵ������C_count_dict.json��
˵������ÿ��������������ͣ�����LabelEncode����ȡֵ0~N����Int���Ͷ�Ӧ�޸ġ�
���ɣ�tr_50_FE.csv, va_50_FE_csv, ts_FE.csv

2.python run.py������ffm�ļ�
��Ҫע��ľ���common.py�ļ��е���������������Ӳ�����18��
��γ���Ҫ���д�����Ҫ��linuxϵͳ�ϣ��õ�cat�����

3.ѵ��
ffm-train -l 0.00001 -k 100 -t 40 -r 1 -s 12  --auto-stop  -p va_50_FE.ffm tr_50_FE.ffm 50_FE.model
ffm-predict ts_FE.csv 50_FE.model 50_FE.output