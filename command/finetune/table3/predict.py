from command import configs
from fairseq.models.xdep import XdepModel

flags = ['O0', 'O1', 'O2', 'O3', 'bcf', 'cff', 'spl', 'sub']
xdep = XdepModel.from_pretrained(f'checkpoints/finetune_table3_all',
                                 checkpoint_file='checkpoint_best.pt',
                                 data_name_or_path=f'data-bin/finetune_table3_all')

xdep.eval()
xdep.cuda(1)


def perf_measure(flag, y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1

    print(f'{flag} TP:{TP}, FP:{FP}, TN:{TN}, FN:{FN}')


for flag in flags:
    samples = {field: [] for field in configs.fields + ['dep_cmp_emb']}
    labels = []

    for field in configs.fields + ['dep_cmp_emb']:
        with open(f'data-src/finetune_table3/table3_{flag}/valid.{field}', 'r') as f:
            for line in f:
                samples[field].append(line.strip())
    with open(f'data-src/finetune_table3/table3_{flag}/valid.label', 'r') as f:
        for line in f:
            labels.append(float(line.strip()))

    predict = []
    for sample_idx in range(len(labels)):
        sample = {field: samples[field][sample_idx] for field in configs.fields + ['dep_cmp_emb']}
        label = labels[sample_idx]

        sample_tokens = xdep.encode(sample)

        logits = xdep.predict('mem_dep', sample_tokens)

        pred = logits.argmax(dim=1)[0].item()
        predict.append(pred)

    perf_measure(flag, labels, predict)
