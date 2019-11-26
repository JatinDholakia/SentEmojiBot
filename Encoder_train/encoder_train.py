import logging
import time

import torch
import torch.nn.functional as F
import torch.optim as optim


from empchat.datasets.loader import TrainEnvironment
from empchat.models import (
    create as create_model,
    load as load_model,
    load_embeddings,
    save as save_model,
    score_candidates
)

from empchat.util import get_logger, get_opt

from argparse import Namespace
opt = Namespace(batch_size=128, bert_add_transformer_layer=False, bert_dim=300, cuda=True, dailydialog_folder=None,
                dataset_name='empchat', dict_max_words=250000, display_iter=100, embeddings='None', embeddings_size=300,
                empchat_folder='ed_datafolder', fasttext=None, fasttext_path=None, fasttext_type=None,
                hits_at_nb_cands=100, learn_embeddings=True, learning_rate=8e-04, load_checkpoint=None,
                log_file='train_save\\model.txt', max_hist_len=4, max_sent_len=100, model='bert', model_dir='train_save',
                model_file='model.pth', model_name='model', n_layers=4, no_shuffle=False, normalize_emb=False,
                normalize_sent_emb=False, num_epochs=10000, optimizer='adamax', pretrained=None, random_seed=92179,
                reactonly=False, reddit_folder='reddit', rm_long_contexts=False, rm_long_sent=False,
                stop_crit_num_epochs=-1, transformer_dim=512, transformer_dropout=0, transformer_n_heads=8)
env = TrainEnvironment(opt) # Making dictionary
dictionary = env.dict
print("Length of dictionary = " + str(len(dictionary["words"])))
print("Embedding Size = "+ str(opt.embeddings_size))

# # env.temp_dict is passed to EmpDataset as dictionary.

opt.transformer_dim = 300
opt.transformer_n_heads=6


opt.model = "transformer"
net = create_model(opt,dictionary["words"]) # Initializes TransformerAdapter
print(net)
# net contains embeddings (dim = len(dictionary),embeddings_size=300)
# ctx_transformer
# cand_transformer

print(env.temp_dict)

# dataset = EmpDataset(
#     splitname,
#     env.temp_dict,
#     data_folder=opt.empchat_folder,
#     maxlen=opt.max_sent_len,
#     reactonly=opt.reactonly,
#     history_len=opt.max_hist_len,
#     fasttext=opt.fasttext,
#     fasttext_type=opt.fasttext_type,
#     fasttext_path=opt.fasttext_path,
# )
# print(dataset)

# print(len(net.ctx_transformer.embeddings.weight))
# print(len(net.ctx_transformer.embeddings.weight[0]))
# print(net.named_parameters)
if opt.embeddings and opt.embeddings != "None":
    load_embeddings(opt, dictionary["words"], net)

# Counting total parameters and trainable parameters
paramnum = 0
trainable = 0
for name, parameter in net.named_parameters():
    if parameter.requires_grad:
        trainable += parameter.numel()
    paramnum += parameter.numel()
print("Total parameters = ", paramnum,"Trainable parameters = ", trainable) # Positional embeddings(1000,300) of ctx_transformer and cand_transformer don't require grad

opt.cuda = False
if opt.cuda:
    print("Using CUDA")
    net = torch.nn.DataParallel(net)
    net = net.cuda()



if opt.optimizer == "adamax":
    lr = opt.learning_rate or 0.002
    named_params_to_optimize = filter(
        lambda p: p[1].requires_grad, net.named_parameters()
    )
    params_to_optimize = (p[1] for p in named_params_to_optimize)
    optimizer = optim.Adamax(params_to_optimize, lr=lr)
    epoch_start = 0
    if(opt.load_checkpoint):
        print("loading checkpoint")
        saved_params = torch.load(opt.load_checkpoint,map_location=lambda storage, loc:storage)
        saved_opt = saved_params['opt']
        word_dict = saved_params['word_dict']
        net = create_model(saved_opt,word_dict)
        net.load_state_dict(saved_params['state_dict'])
        optimizer = optimizer.load_state_dict(saved_params['optim_dict'])
        epoch_start = saved_params['epoch']
else:
    lr = opt.learning_rate or 0.01
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, net.parameters()), lr=lr
    )

print("Running first validation")
from retrieval_train import validate, train, loss_fn
start_time = time.time()
best_loss = float("+inf")
test_data_shuffled = env.build_valid_dataloader(True) # Only for shuffled data
import numpy as np
with torch.no_grad():
    validate(env,
        opt,
        0,
        net,
        test_data_shuffled,
        nb_candidates=opt.hits_at_nb_cands,
        shuffled_str="shuffled",

    )

# print("Started Training")
# train_data = None
# for epoch in range(epoch_start, opt.num_epochs):
#     if train_data is None or opt.dataset_name == "reddit":
#         train_data = env.build_train_dataloader(epoch)
#         print("Train data loaded")
#     print("Epoch {}".format(epoch))
#     train(epoch, start_time, net, optimizer, opt, train_data)
#     # print("Training completed for epoch {}".format(epoch))
#     with torch.no_grad():
#         # We compute the loss both for shuffled and not shuffled case.
#         # however, the loss that determines if the model is better is the
#         # same as the one used for training.
#         loss_shuffled = validate(
#             opt,
#             epoch,
#             net,
#             test_data_shuffled,
#             nb_candidates=opt.hits_at_nb_cands,
#             shuffled_str="shuffled",
#         )
#         # loss_not_shuffled = validate(
#         #     epoch,
#         #     net,
#         #     test_data_not_shuffled,
#         #     nb_candidates=opt.hits_at_nb_cands,
#         #     shuffled_str="not-shuffled",
#         # )
#         if opt.no_shuffle:
#             loss = loss_not_shuffled
#         else:
#             loss = loss_shuffled
#         print("Loss = ", str(loss))
#         if('losses.csv' in os.listdir()):
#             with open('losses.csv','a') as f:
#                 f.write(str(loss))
#         else:
#             with open('losses.csv','w') as f:
#                 f.write(str(loss))
#         if loss < best_loss:
#             best_loss = loss
#             best_loss_epoch = epoch
#             print("New best loss, saving model to {}".format(opt.model_file))
#             logging.info(f"New best loss, saving model to {opt.model_file}")
#             save_model(opt.model_file, net, dictionary, optimizer,epoch)
#         # Stop if it's been too many epochs since the loss has decreased
#         if opt.stop_crit_num_epochs != -1:
#             if epoch - best_loss_epoch >= opt.stop_crit_num_epochs:
#                 break