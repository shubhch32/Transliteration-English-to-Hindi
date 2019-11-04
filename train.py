import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.utils import shuffle
from tensorflow.python.layers import core as layers_core
import math
import argparse


def get_args():
    '''This function parses and return arguments passed in'''
    # Assign description to the help doc
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument(
        '--lr',type=float, required=True)
    parser.add_argument(
        '--batch_size',type=int,required=True)
    parser.add_argument(
        '--init',type=int,required=True)
    parser.add_argument(
        '--dropout_prob',type=float, required=True)
    parser.add_argument(
        '--decode_method',type=int, required=True)
    parser.add_argument(
        '--beam_width',type=int, required=True)     
    parser.add_argument(
        '--save_dir',type=str,required=True)
    parser.add_argument(
        '--epochs',type=int,required=True)
    parser.add_argument(
        '--train',type=str,required=True)
    parser.add_argument(
        '--val',type=str,required=True)
    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    lr = args.lr
    batch_size=args.batch_size
    init=args.init
    dropout_prob=args.dropout_prob
    decode_method=args.decode_method
    beam_width=args.beam_width
    save_dir=args.save_dir
    epochs=args.epochs
    train=args.train
    val=args.val
    # Return all variable values
    return lr,batch_size,init,dropout_prob,decode_method,beam_width,save_dir,epochs,train,val


lr,batch_size,init,dropout_prob,decode_method,beam_width,save_dir,epochs,train,val = get_args()

train=pd.read_csv(train)
# test=pd.read_csv('test_final.csv')
valid=pd.read_csv(val)
train = train.drop('id',axis=1)
valid=valid.drop('id',axis=1)
# test = test.drop('id',axis=1)
# train = "C:/Users/ShubhamCh/Desktop/Studies/8th Sem/Deep Learning/PA3/train.csv"
# test = "C:/Users/ShubhamCh/Desktop/Studies/8th Sem/Deep Learning/PA3/partial_test_400.csv"
# val = "C:/Users/ShubhamCh/Desktop/Studies/8th Sem/Deep Learning/PA3/valid.csv"
# train = pd.read_csv(train.csv)
# valid = pd.read_csv(val.csv)
# test = pd.read_csv(test.csv)
# train = train.drop('id',axis=1)
# valid = valid.drop('id',axis=1)
# test = test.drop('id',axis=1)
train_x = np.array(train['ENG'])
train_y = np.array(train['HIN'])
valid_x = np.array(valid['ENG'])
valid_y = np.array(valid['HIN'])
# test_x = np.array(test['ENG'])


x_train=np.copy(train_x)
y_train=np.copy(train_y)

np.random.seed(1234)
tf.random.set_random_seed(1234)
learning_rate=lr
training_iters=epochs
num_layers = 2
max_size=65
lamb = 0.01
keep_prob=1.0-dropout_prob
dict_en={} 
dict_rev_en={}
i_x=0
for x in x_train:
    for c in x:
        if(c==' '):
            continue
        if c in dict_en:
            #dict_en[c]+=1
            continue
        else:
            dict_en[c]=i_x
            dict_rev_en[i_x]=c
            i_x+=1

dict_en["epad"]=i_x
dict_rev_en[i_x]="epad"
i_x+=1
dict_en["unknown"]=i_x
dict_rev_en[i_x]="unknown"
print(len(dict_en))
dict_hn={}
dict_rev_hn={}
dict_hn["start"]=0
dict_hn["stop"]=1
dict_rev_hn[0]="start"
dict_rev_hn[1]="stop"
i_y=2
for y in y_train:
    for c in y:
        if(c==' '):
            continue
        if c in dict_hn:
            #dict_hn[c]+=1
            continue
        else:
            dict_hn[c]=i_y
            dict_rev_hn[i_y]=c
            i_y+=1
dict_hn["hpad"]=i_y
dict_rev_hn[i_y]="hpad"
i_y+=1
dict_hn["unknown"]=i_y
dict_rev_hn[i_y]="unknown"
print(len(dict_hn))
def daprox(x_train,dict_en):
    ven=len(dict_en)
    x_train_oh=[]
    x_train_num=[]
    for x in x_train:
        ohx=[]
        nx=[]
        for c in x:
            if(c==' '):
                continue
            if c in dict_en:
                k=c
            else:
                k="unknown"
            zer=np.zeros((1,ven))[0]
            zer[dict_en[k]]=1
            ohx.append(zer)
            nx.append(dict_en[k])
        lohx=len(ohx)
        lnx=len(nx)
        zer=np.zeros((1,ven))[0]
        zer[i_x-1]=1
        for i in range(max_size-lnx):
            ohx.append(zer)
            nx.append(i_x-1)
        x_train_oh.append((ohx))
        nx=np.asarray(nx)
        x_train_num.append((nx))
    x_train_num=np.transpose(np.asarray(x_train_num))
    return x_train_num,x_train_oh

def daproy(y_train,dict_hn):
    vhn=len(dict_hn)
    y_train_oh=[]
    y_train_num=[]
    y_train_num1=[]
    for y in y_train:
        ohy=[]
        ny=[]
        ny1=[]
        ny.append(dict_hn["start"])
        for c in y:
            if(c==' '):
                continue
            if c in dict_hn:
                k=c
            else:
                k="unknown"
            zer=np.zeros((1,vhn))[0]
            zer[dict_hn[k]]=1
            ohy.append(zer)
            ny.append(dict_hn[k])
            ny1.append(dict_hn[k])
        zer=np.zeros((1,vhn))[0]
        zer[dict_hn["stop"]]=1
        ohy.append(zer)
        ny.append(dict_hn["stop"])
        ny1.append(dict_hn["stop"])
        lohy=len(ohy)
        lny=len(ny)
        zer=np.zeros((1,vhn))[0]
        zer[i_y-1]=1
        for i in range(max_size-lny):
            ohy.append(zer)
            ny.append(i_y-1)
            ny1.append(i_y-1)
        ohy.append(zer)
        ny1.append(i_y-1)
        y_train_oh.append((ohy))
        y_train_num.append((ny))
        y_train_num1.append((ny1))
        
    y_train_num=np.transpose(np.asarray(y_train_num))
    y_train_num1=np.transpose(np.asarray(y_train_num1))
    return y_train_num,y_train_num1,y_train_oh

x_train_num,x_train_oh=daprox(x_train,dict_en)
y_train_num,y_train_num1,y_train_oh=daproy(y_train,dict_hn)

x_valid_num,x_valid_oh=daprox(valid_x,dict_en)
y_valid_num,y_valid_num1,y_valid_oh=daproy(valid_y,dict_hn)

# x_test_num,x_test_oh=daprox(test_x,dict_en)

print(len(x_train_oh))
print(len(y_train_oh))


tf.reset_default_graph()
x_oh = tf.placeholder(tf.int32, [None,max_size, 45])
y_oh = tf.placeholder(tf.int32, [None,max_size, 87])
x_num = tf.placeholder(tf.int32, [max_size,None])
y_num = tf.placeholder(tf.int32, [max_size,None])
y_num1 = tf.placeholder(tf.int32, [max_size,None])
decoder_lengths = tf.placeholder(tf.int32, shape=[None], name="decoder_length")
bs = tf.shape(x_num)[1]

weights={}
if(init==1):
    weights = {
        'winem': tf.get_variable('W0', shape=(45,256), initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None)),
        'woutem': tf.get_variable('W1', shape=(87,256), initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None)),
    }
else:
    weights = {
        'winem': tf.get_variable('W0', shape=(45,256), initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0)),
        'woutem': tf.get_variable('W1', shape=(87,256), initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0)),
    }

def inembed(x,weights):
    print(x.shape)
    inemb=tf.nn.embedding_lookup(weights['winem'], x)
    print(inemb.shape)
    return inemb

def encoder(x):
    lstmUnits = 256
    print(x[0].shape)
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(lstmUnits, forget_bias=1.0, state_is_tuple=True)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(lstmUnits, forget_bias=1.0, state_is_tuple=True)
    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, keep_prob)
    lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, keep_prob)
    (out_fw, out_bw), (state_fw, state_bw)  = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell, inputs=x, time_major=True, dtype=tf.float32)
    bi_state_c = tf.concat((state_fw.c, state_bw.c), 1)
    bi_state_h = tf.concat((state_fw.h, state_bw.h), 1)
    bi_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(c=bi_state_c, h=bi_state_h)
    state = bi_lstm_state
    output = tf.concat((out_fw, out_bw), 2)
    return output,state

inemb=inembed(x_num,weights)
output_enc,enc_state=encoder(inemb)
print("Done 2")
print(enc_state[0].shape)
print("Done 3")

#--------------------------------------
#Decoder_bhai

outemb=tf.nn.embedding_lookup(weights['woutem'], y_num)
print(outemb.shape)
projection_layer = layers_core.Dense(87, use_bias=False)
helper = tf.contrib.seq2seq.TrainingHelper(outemb, decoder_lengths, time_major=True)
# decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(512)
decoder_cell=[]
for i in range(num_layers):
    cell = tf.contrib.rnn.LSTMCell(512)
    cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
    decoder_cell.append(cell)

attention_states = tf.transpose(output_enc, [1, 0, 2])
print(attention_states.shape)
print("Done 5")
attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(512, attention_states,memory_sequence_length=None)
print("Done 6")
# decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,attention_layer_size=512)
decoder_cell[0] = tf.contrib.seq2seq.AttentionWrapper(decoder_cell[0], attention_mechanism,attention_layer_size=512)
print("Done 7")
initial_state = [enc_state for i in range(num_layers)]
# initial_state = decoder_cell.zero_state(bs, tf.float32).clone(cell_state=enc_state)
cell_state = decoder_cell[0].zero_state(dtype=tf.float32, batch_size = bs)
initial_state[0] = cell_state.clone(cell_state=initial_state[0])
initial_state = tuple(initial_state)
decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_cell)
print("Done 8")
decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state,output_layer=projection_layer)
tgt_sos_id=0
tgt_eos_id=1
inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(weights['woutem'],tf.fill([bs], tgt_sos_id), tgt_eos_id)
inference_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, inference_helper, initial_state,output_layer=projection_layer)
maximum_iterations = max_size
outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder, maximum_iterations=maximum_iterations)
translations = outputs.sample_id
final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder)

#---------------------------------------------

#--------------------------------------------------
#Beam search



# start_tokens = [tgt_sos_id for i in range(batch_size)]
# decoder_initial_state = tf.contrib.seq2seq.tile_batch(initial_state, multiplier=beam_width)
# decoder1 = tf.contrib.seq2seq.BeamSearchDecoder(
#         cell=decoder_cell,
#         embedding=weights['woutem'],
#         start_tokens=start_tokens,
#         end_token=tgt_eos_id,
#         initial_state=decoder_initial_state,
#         beam_width=beam_width,
#         output_layer=projection_layer,
#         length_penalty_weight=0.0,
#         coverage_penalty_weight=0.0)
# outputs1, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder1, maximum_iterations=maximum_iterations)
# translations1 = outputs1.predicted_ids
#------------------------------------------------------


logits = final_outputs.rnn_output
logits1 = outputs.rnn_output
print("jj",logits1.shape)
print("kk",logits.shape)
# cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.transpose(y_num1), logits=logits))
# regularizer = tf.add_n([tf.nn.l2_loss(weights['winem']),tf.nn.l2_loss(weights['woutem'])])
# cost = tf.add_n([tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_oh, logits=logits)) , lamb * regularizer])
cost = tf.contrib.seq2seq.sequence_loss(logits=logits,targets=tf.transpose(y_num1),weights=tf.ones([bs, max_size]),average_across_timesteps=True,average_across_batch=True,softmax_loss_function=None,name=None)
#cost1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_oh, logits=logits1))
cost1 = tf.contrib.seq2seq.sequence_loss(logits=logits1,targets=tf.transpose(y_num1),weights=tf.ones([bs, max_size]),average_across_timesteps=True,average_across_batch=True,softmax_loss_function=None,name=None)
# loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_labels, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
sess=tf.Session()
sess.run(tf.global_variables_initializer()) 
train_loss = []
valid_loss = []
train_accuracy = []
test_accuracy = []
train_acc = 0.0
saver = tf.train.Saver(max_to_keep=training_iters)
save_dir = "PA2/"
train_loss=[]
valid_loss=[]
accu=0.0
max_acc=0.0
p=0
for i in range(training_iters):
    t_loss,t_val = [],[]
    for batch in range(math.ceil(len(x_train_num[0])/batch_size)):
        batch_x = x_train_num[:,batch*batch_size:min((batch+1)*batch_size,len(x_train_num[0]))]
        batch_y = y_train_num[:,batch*batch_size:min((batch+1)*batch_size,len(y_train_num[0]))]
        batch_y1 = y_train_num1[:,batch*batch_size:min((batch+1)*batch_size,len(y_train_num1[0]))]
        y_train_oh1 = y_train_oh[batch*batch_size:min((batch+1)*batch_size,len(y_train_oh))]
        just_store=np.ones(batch_x.shape[1], dtype=int) * max_size
        opt = sess.run(optimizer, feed_dict={x_num:batch_x,y_num:batch_y,y_num1:batch_y1,decoder_lengths:just_store,y_oh:y_train_oh1})

    for batch in range(math.ceil(len(x_train_num[0])/batch_size)):
        batch_x = x_train_num[:,batch*batch_size:min((batch+1)*batch_size,len(x_train_num[0]))]
        batch_y = y_train_num[:,batch*batch_size:min((batch+1)*batch_size,len(y_train_num[0]))]
        batch_y1 = y_train_num1[:,batch*batch_size:min((batch+1)*batch_size,len(y_train_num1[0]))]
        y_train_oh1 = y_train_oh[batch*batch_size:min((batch+1)*batch_size,len(y_train_oh))]
        just_store=np.ones(batch_x.shape[1], dtype=int) * max_size
        loss =  sess.run(cost, feed_dict={x_num:batch_x,y_num:batch_y,y_num1:batch_y1,decoder_lengths:just_store,y_oh:y_train_oh1})
        t_loss.append(loss)

    for batch in range(math.ceil(len(x_valid_num[0])/batch_size)):
        batch_x = x_valid_num[:,batch*batch_size:min((batch+1)*batch_size,len(x_valid_num[0]))]
        batch_y = y_valid_num[:,batch*batch_size:min((batch+1)*batch_size,len(y_valid_num[0]))]
        batch_y1 = y_valid_num1[:,batch*batch_size:min((batch+1)*batch_size,len(y_valid_num1[0]))]
        y_valid_oh1 = y_valid_oh[batch*batch_size:min((batch+1)*batch_size,len(y_valid_oh))]
        just_store=np.ones(batch_x.shape[1], dtype=int) * max_size
        loss1 =  sess.run(cost, feed_dict={x_num:batch_x,y_num:batch_y,y_num1:batch_y1,decoder_lengths:just_store,y_oh:y_valid_oh1})
        t_val.append(loss1)
#         t_acc.append(acc)

    loss=sum(t_loss)/len(t_loss)
    loss1=sum(t_val)/len(t_val)
    pred1=[]
    for batch in range(math.ceil(len(x_valid_num[0])/batch_size)):
        batch_x = x_valid_num[:,batch*batch_size:min((batch+1)*batch_size,len(x_valid_num[0]))]
        test_pred1 = sess.run(translations ,feed_dict={x_num:batch_x})
        #   print(test_pred1.shape)
        for ii in range(test_pred1.shape[0]):
            s=""
            for jj in test_pred1[ii]:
                c = dict_rev_hn[jj]
                if c=="stop":
                    break
                else:
                    s=s+c+" "
            s=str.strip(s)
            pred1.append(s)
            #     print(pred1[i])
    count = 0.0
    for ii in range(len(pred1)):
        if valid_y[ii]==pred1[ii]:
            count+=1

    acc = 100*count/len(pred1)
    if(acc>max_acc):
        max_acc=acc
        p=0
    else:
        p+=1
        if(p>=5):
            break
    if accu>acc:
        saver.restore(sess,save_dir+"model"+str(i-1)+".ckpt")
        learning_rate = learning_rate/2.0
    else:
        accu=acc
        train_loss.append(loss)
        valid_loss.append(loss1)
    save_path = saver.save(sess,save_dir+"model"+str(i)+".ckpt")
    print("Accuracy = "+str(acc))
#     acc=sum(t_acc)/len(t_acc)
    print("Iter " + str(i) + ", Training Loss= " + "{:.4f}".format(loss) + ", Validation Loss= " + "{:.4f}".format(loss1))# + ", Training Accuracy= " + \"{:.4f}".format(acc))
    print("Optimization Finished!")

#     pred=[]
# #     print(x_test_num.shape)
#     for batch in range(math.ceil(len(x_test_num[0])/batch_size)):
#         batch_x = x_test_num[:,batch*batch_size:min((batch+1)*batch_size,len(x_test_num[0]))]
#         test_pred1 = sess.run(translations ,feed_dict={x_num:batch_x})
# #         print(test_pred1.shape)

#         for ii in range(test_pred1.shape[0]):
#             s=""
#             for jj in test_pred1[ii]:
#                 c = dict_rev_hn[jj]
#                 if c=="start":
#                     continue
#                 if c=="stop":
#                     break
#                 else:
#                     s=s+c+" "
#             s=str.strip(s)
#             pred.append(s)

#     y_pred_test = []
#     for ii in range (len(pred)):
#         temp = [ii,pred[ii]]
#         y_pred_test.append(temp)
#     df = pd.DataFrame(y_pred_test,columns=['id','HIN'])

#     df.to_csv('df'+str(i)+'.csv',index=False)

#--------------Beam Inference

# pred1=[]
# for batch in range(math.ceil(len(x_valid_num[0])//batch_size)):
#     batch_x = x_valid_num[:,batch*batch_size:min((batch+1)*batch_size,len(x_valid_num[0]))]
#     test_pred1 = sess.run(translations1 ,feed_dict={drop:1,x_num:batch_x})
#     print(test_pred1.shape)
#     for ii in range(test_pred1.shape[0]):
#         s=""
#         for jj in range(test_pred1.shape[1]):
#             c1 = test_pred1[ii][:,0:1]
#             c=dict_rev_hn[c1[jj][0]]
#             if c=="stop":
#                 break
#             else:
#                 s=s+c+" "
#         s=str.strip(s)
#     pred1.append(s)
# count = 0.0
# for ii in range(len(pred1)):
#     if valid_y[ii]==pred1[ii]:
#         count+=1

# acc = 100*count/len(pred1)
# print(acc)
#-----------------------------------
ep=len(train_loss)
e=1+np.arange(ep)
plt.plot(e,train_loss)
plt.plot(e,valid_loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(("Training Loss","Validation Loss"))
plt.title('Negative Log Likelihood')
plt.savefig("Fig1")