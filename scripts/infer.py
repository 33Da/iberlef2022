import torch
import torch.nn as nn
from torch import optim
from model import JointEntityRelation, MultiTaskLossWrapper,JointEntityRelation_fourhidden
import json
from train import Train
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
BATCH_STATUS=128
EPOCH=10
BATCH_SIZE=1
PRETRAINED_MODEL = 'beto'
EARLY_STOP = 15
LEARNING_RATE=2e-5
trainset_name='training_develop'

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("加载数据")
    if PRETRAINED_MODEL == 'beto':
        trainset = json.load(open(r'../data/preprocessed/trainset.json'))
        # devset = json.load(open('../data/preprocessed2/ '+ filename + ".json"))
        devset = json.load(open("../data/preprocessed/devset.json"))
        model = JointEntityRelation(pretrained_model_path='dccuchile/bert-base-spanish-wwm-cased')
    elif PRETRAINED_MODEL == 'mt5':
        trainset = json.load(open('data/original/ref/'+trainset_name+'/input_mt5.json'))
        devset = json.load(open('data/original/ref/develop/input_mt5.json'))
        model = JointEntityRelation(pretrained_model_path='/scratch/thiago.ferreira/mt5')
    else:
        trainset = json.load(open('data/original/ref/'+trainset_name+'/input_multilingual.json'))
        devset = json.load(open('data/original/ref/develop/input_multilingual.json'))
        model = JointEntityRelation(pretrained_model_path='bert-base-multilingual-cased')
    model.to(device)

    criterion = nn.NLLLoss()

    initial_lr = LEARNING_RATE / 10
    lmbda = lambda epoch: min(10, epoch + 1)
    write_path = 'static_dict/model1.pt'
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lmbda)


    model = torch.load(write_path)
    model.eval()

    loss_func = MultiTaskLossWrapper(1)
    loss_func.to(device)
    loss_optimizer = optim.AdamW(loss_func.parameters(), lr=LEARNING_RATE)

    trainer = Train(model, criterion, optimizer, scheduler, trainset, devset, EPOCH, BATCH_SIZE, early_stop=EARLY_STOP,
                    write_path='', pretrained_model=PRETRAINED_MODEL, batch_status=BATCH_STATUS, task='entity',
                    loss_func=loss_func, loss_optimizer=loss_optimizer)

    for fname in Path("../data/2022/original/test_background/file3/text-files2/").rglob("*.txt"):
        print(fname)
        filename = fname.name
        filename = filename.split(".txt")[0]
        print(filename)

        # print("开始训练")
        # trainer.train()

        # print("加载模型")
        # model = torch.load(write_path)
        # initial_lr = LEARNING_RATE / 10
        # optimizer = optim.AdamW(model.parameters(), lr=initial_lr)
        # lmbda = lambda epoch: min(10, epoch + 1)
        # scheduler = LambdaLR(optimizer, lr_lambda=lmbda)
        # write_path = 'static_dict/model.pt'
        # trainer = Train(model, criterion, optimizer, scheduler, trainset, devset, EPOCH, BATCH_SIZE, early_stop=EARLY_STOP, write_path=write_path, pretrained_model=PRETRAINED_MODEL, batch_status=BATCH_STATUS, task='entity')
        # trainer.train()

        # loss_func = MultiTaskLossWrapper(1)
        # loss_func.to(device)
        # loss_optimizer = optim.AdamW(loss_func.parameters(), lr=LEARNING_RATE)

        # write_path = '/scratch/thiago.ferreira/vicom_final/model.pt'
        # EPOCH=100
        # model = torch.load(write_path)
        # initial_lr = LEARNING_RATE / 10
        # optimizer = optim.AdamW(model.parameters(), lr=initial_lr)
        # lmbda = lambda epoch: min(10, epoch + 1)
        # scheduler = LambdaLR(optimizer, lr_lambda=lmbda)
        # write_path = 'static_dict/model.pt'
        #     trainer = Train(model, criterion, optimizer, scheduler, trainset, devset, EPOCH, BATCH_SIZE, early_stop=EARLY_STOP, write_path=write_path, pretrained_model=PRETRAINED_MODEL, batch_status=BATCH_STATUS, task='entity+relation', loss_func=loss_func, loss_optimizer=loss_optimizer)
        #     trainer.train()

        # EVALUATION
        # loss_func = MultiTaskLossWrapper(2)
        # loss_func.to(device)
        # loss_optimizer = optim.AdamW(loss_func.parameters(), lr=LEARNING_RATE)

        # model = torch.load(write_path)
        # model.eval()
        # initial_lr = LEARNING_RATE / 10
        # optimizer = optim.AdamW(model.parameters(), lr=initial_lr)
        # lmbda = lambda epoch: min(10, epoch + 1)
        # scheduler = LambdaLR(optimizer, lr_lambda=lmbda)
        # trainer = Train(model, criterion, optimizer, scheduler, trainset, devset, EPOCH, BATCH_SIZE, early_stop=EARLY_STOP, write_path='', pretrained_model=PRETRAINED_MODEL, batch_status=BATCH_STATUS, task='entity', loss_func=loss_func, loss_optimizer=loss_optimizer)

        # print("################ training")
        # trainer.eval_mode = 'training'
        # trainer.eval_class_report()
        # trainer.eval()


        # print("################ develop")
        # trainer.eval_mode = 'develop'
        # trainer.eval_class_report()
        # trainer.eval("devset.json")

        print("################ testing")
        trainer.eval_mode = 'testing'
        trainer.eval_class_report()
        trainer.eval(filename)