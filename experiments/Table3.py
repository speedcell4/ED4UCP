import os

import pandas as pd

from KimEval import sentence_level_f1
from experiments.constants import GOLD_NAME
from experiments.constants import MBR_GEN_PATH
from experiments.constants import MBR_SEL_PATH
from experiments.constants import MOTHER_PATH
from experiments.constants import ORACLE
from experiments.constants import RNNG_SUSANNE_FILE_NAME
from experiments.constants import RUN
from experiments.constants import SUSANNE_FILE_NAME
from experiments.constants import TABLE6_GUIDE_NAME
from experiments.constants import URNNG_SUSANNE_FILE_NAME
from library.ensemble import ensemble
from library.heuristics import left
from library.heuristics import oracle
from library.heuristics import right


def table_3():
    golds = open(os.path.join(MOTHER_PATH, GOLD_NAME, SUSANNE_FILE_NAME)).readlines()
    print('=' * 60)
    print('   Model\t\t\t| Run1\t +RNNG\t +URNNG')
    print('-' * 60)
    print(f' 1  Left branching\t\t| {sentence_level_f1(golds, map(left, golds))}')
    print(f' 2  Right branching\t\t| {sentence_level_f1(golds, map(right, golds))}')
    print('-' * 60)

    guide = pd.read_csv(os.path.join(MOTHER_PATH, TABLE6_GUIDE_NAME)).set_index(RUN)
    run1 = guide.index[0]

    references = []
    for i, model in enumerate(guide.columns):
        parsed = open(os.path.join(MOTHER_PATH, model, guide.loc[run1, model], SUSANNE_FILE_NAME)).readlines()
        references.append(parsed)
        run1_f1 = sentence_level_f1(golds, parsed)
        rnng_f1 = sentence_level_f1(golds, open(
            os.path.join(MOTHER_PATH, model, guide.loc[run1, model], RNNG_SUSANNE_FILE_NAME)).readlines())
        urnng_f1 = sentence_level_f1(golds, open(
            os.path.join(MOTHER_PATH, model, guide.loc[run1, model], URNNG_SUSANNE_FILE_NAME)).readlines())
        print(f' {i + 3}  {model}\t\t' + ("\t" if len(model) < 12 else "") + f'| {run1_f1}\t {rnng_f1}\t {urnng_f1}')
    print('-' * 60)

    run1_f1 = sentence_level_f1(golds, ensemble(references, MBR_mode='selective'))
    rnng_f1 = sentence_level_f1(golds, open(
        os.path.join(MOTHER_PATH, MBR_SEL_PATH, RUN + str(run1), RNNG_SUSANNE_FILE_NAME)).readlines())
    urnng_f1 = sentence_level_f1(golds, open(
        os.path.join(MOTHER_PATH, MBR_SEL_PATH, RUN + str(run1), URNNG_SUSANNE_FILE_NAME)).readlines())
    print(f' {len(guide.columns) + 3} {MBR_SEL_PATH}\t\t' + (
        "\t" if len(MBR_SEL_PATH) < 12 else "") + f'| {run1_f1}\t {rnng_f1}\t {urnng_f1}')

    run1_f1 = sentence_level_f1(golds, ensemble(references))
    rnng_f1 = sentence_level_f1(golds, open(
        os.path.join(MOTHER_PATH, MBR_GEN_PATH, RUN + str(run1), RNNG_SUSANNE_FILE_NAME)).readlines())
    urnng_f1 = sentence_level_f1(golds, open(
        os.path.join(MOTHER_PATH, MBR_GEN_PATH, RUN + str(run1), URNNG_SUSANNE_FILE_NAME)).readlines())
    print(f' {len(guide.columns) + 4} {MBR_GEN_PATH}\t\t' + (
        "\t" if len(MBR_GEN_PATH) < 12 else "") + f'| {run1_f1}\t {rnng_f1}\t {urnng_f1}')
    print('-' * 60)

    rnng_f1 = sentence_level_f1(golds, open(os.path.join(MOTHER_PATH, ORACLE, RNNG_SUSANNE_FILE_NAME)).readlines())
    urnng_f1 = sentence_level_f1(golds, open(os.path.join(MOTHER_PATH, ORACLE, URNNG_SUSANNE_FILE_NAME)).readlines())
    print(
        f' {len(guide.columns) + 5} {ORACLE}\t\t' + ("\t" if len(ORACLE) < 12 else "") + f'| \t {rnng_f1}\t {urnng_f1}')

    print(f' {len(guide.columns) + 6} SUSANNE Oracle\t\t| {sentence_level_f1(golds, map(oracle, golds))}')
    print('=' * 60)
