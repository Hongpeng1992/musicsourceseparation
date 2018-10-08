"""

Modifications to trainCNN.py at DeepConvSep -----> translation from Lasagne, Theano to Keras, Tensorflow

"""

import os,sys
import transform
from transform import transformFFT
import dataset
from dataset import LargeDataset
import util

import numpy as np
import re
from scipy.signal import blackmanharris as blackmanharris
import shutil
import time
import cPickle
import re
import climate
import ConfigParser

import tensorflow as tf

#import theano.sandbox.rng_mrg

import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2DTranspose, Reshape, ReLU
from keras.layers import Convolution2D as Conv2D
from keras.optimizers import SGD

logging = climate.get_logger('trainer')

climate.enable_default_logging()


def load_model(filename):
    f=file(filename,'rb')
    params=cPickle.load(f)
    f.close()
    return params

def save_model(filename, model):
    params=lasagne.layers.get_all_param_values(model) ################
    f = file(filename, 'wb')
    cPickle.dump(params,f,protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    return None


def build_ca(input_var=None, batch_size=32,time_context=30,feat_size=513):

        """
    Builds a network with TF
    
    Parameters
    ----------
    input_var : TF tensor
        The input for the network
    batch_size : int, optional
        The number of examples in a batch   
    time_context : int, optional
        The time context modeled by the network. 
    feat_size : int, optional
        The feature size modeled by the network (last dimension of the feature vector)
    Yields
    ------
    l_out : TF tensor
        The output of the network
    """
    ## ENCODING STAGE

    main_input=Input(shape=(batch_size,1,time_context,feat_size))

    vertical_conv=Conv2D(50, (1,feat_size), strides=(1,1), padding='valid', use_bias=True)(main_input)
    horiz_conv=Conv2D(50, (int(time_context/2),1), strides=(1,1), padding='valid', use_bias=True)(vertical_conv)
    dense_gen=Dense(128, activation='relu')(horiz_conv)


    ## DECODING STAGE
     

    dense_1=Dense(horiz_conv.output_shape[1]*horiz_conv.output_shape[2]*horiz_conv.output_shape[3], activation='relu')(dense_gen)
    reshape_1=Reshape(batch_size,l_conv2.output_shape[1],l_conv2.output_shape[2], l_conv2.output_shape[3])(dense_1)
    inv_hor_1=Conv2DTranspose(50, (int(time_context/2),1), strides=(1,1), padding='valid')(reshape_1)
    inv_ver_1=Conv2DTranspose(50, (1,feat_size), strides=(1,1), padding='valid')(inv_hor_1)

    dense_2=Dense(horiz_conv.output_shape[1]*horiz_conv.output_shape[2]*horiz_conv.output_shape[3], activation='relu')(dense_gen)
    reshape_2=Reshape(batch_size,l_conv2.output_shape[1],l_conv2.output_shape[2], l_conv2.output_shape[3])(dense_2)
    inv_hor_2=Conv2DTranspose(50, (int(time_context/2),1), strides=(1,1), padding='valid')(reshape_2)
    inv_ver_2=Conv2DTranspose(50, (1,feat_size), strides=(1,1), padding='valid')(inv_hor_2)

    dense_3=Dense(horiz_conv.output_shape[1]*horiz_conv.output_shape[2]*horiz_conv.output_shape[3], activation='relu')(dense_gen)
    reshape_3=Reshape(batch_size,l_conv2.output_shape[1],l_conv2.output_shape[2], l_conv2.output_shape[3])(dense_3)
    inv_hor_3=Conv2DTranspose(50, (int(time_context/2),1), strides=(1,1), padding='valid')(reshape_3)
    inv_ver_3=Conv2DTranspose(50, (1,feat_size), strides=(1,1), padding='valid')(inv_hor_3)

    dense_4=Dense(horiz_conv.output_shape[1]*horiz_conv.output_shape[2]*horiz_conv.output_shape[3], activation='relu')(dense_gen)
    reshape_4=Reshape(batch_size,l_conv2.output_shape[1],l_conv2.output_shape[2], l_conv2.output_shape[3])(dense_4)
    inv_hor_4=Conv2DTranspose(50, (int(time_context/2),1), strides=(1,1), padding='valid')(reshape_4)
    inv_ver_4=Conv2DTranspose(50, (1,feat_size), strides=(1,1), padding='valid')(inv_hor_4)

    merged=Concatenate([inv_ver_1,inv_ver_2,inv_ver_3,inv_ver_4], axis=1, use_bias=True) #### Comprobar si Concatenate admite use_bias

    #main_out = lasagne.layers.NonlinearityLayer(lasagne.layers.BiasLayer(l_merge), nonlinearity=lasagne.nonlinearities.rectify)
    main_out=ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(merged)

    return main_out

model = Model(inputs=input_var, outputs=main_out)

model.summary()

"""
    model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True), metrics=['accuracy'])

    history = model.fit(X_train, y_train,
             epochs=100,
             batch_size=128,
             validation_data=(X_test, y_test),
             verbose=1)
"""

def train_auto(train,fun,transform,testdir,outdir,num_epochs=30,model="1.pkl",scale_factor=0.3,load=False,skip_train=False,skip_sep=False):
    """
    Trains a network built with \"fun\" with the data generated with \"train\"
    and then separates the files in \"testdir\",writing the result in \"outdir\"

    Parameters
    ----------
    train : Callable, e.g. LargeDataset object
        The callable which generates training data for the network: inputs, target = train()
    fun : KERAS network object, TF tensor
        The network to be trained  
    transform : transformFFT object
        The Transform object which was used to compute the features (see compute_features.py)
    testdir : string, optional
        The directory where the files to be separated are located
    outdir : string, optional
        The directory where to write the separated files
    num_epochs : int, optional
        The number the epochs to train for (one epoch is when all examples in the dataset are seen by the network)
    model : string, optional
        The path where to save the trained model (theano tensor containing the network) 
    scale_factor : float, optional
        Scale the magnitude of the files to be separated with this factor
    Yields
    ------
    losser : list
        The losses for each epoch, stored in a list
    """

    logging.info("Building Autoencoder")
    input_var2 = tf.placeholder('inputs')
    target_var2 = tf.placeholder('targets')
    rand_num = tf.Variable('rand_num')
    
    eps=1e-8
    alpha=0.001
    beta=0.01
    beta_voc=0.03


   ## Investigar la funcion fun (aparece en dataset.py pero con 3 argumentos, no 4...) Yo diría que hay que montar la red neuronal con build_ca
    network2 = fun(input_var=input_var2,batch_size=train.batch_size,time_context=train.time_context,feat_size=train.input_size) ###build_ca
    
    if load:
        params=load_model(model)
        lasagne.layers.set_all_param_values(network2,params) ## keras.layers(set_weights(params))

    prediction2 = lasagne.layers.get_output(network2, deterministic=True)

    rand_num = np.random.uniform(size=(train.batch_size,1,train.time_context,train.input_size))

    voc=prediction2[:,0:1,:,:]+eps*rand_num
    bas=prediction2[:,1:2,:,:]+eps*rand_num
    dru=prediction2[:,2:3,:,:]+eps*rand_num
    oth=prediction2[:,3:4,:,:]+eps*rand_num

    mask1=voc/(voc+bas+dru+oth)
    mask2=bas/(voc+bas+dru+oth)
    mask3=dru/(voc+bas+dru+oth)
    mask4=oth/(voc+bas+dru+oth)

    vocals=mask1*input_var2
    bass=mask2*input_var2
    drums=mask3*input_var2
    others=mask4*input_var2

    # Calculo de diferencias entre fuentes para entrenar 

    train_loss_recon_vocals = keras.losses.mean_squared_error(vocals,target_var2[:,0:1,:,:])
    alpha_component = alpha*keras.losses.mean_squared_error(vocals,target_var2[:,1:2,:,:])
    alpha_component += alpha*keras.losses.mean_squared_error(vocals,target_var2[:,2:3,:,:])    
    train_loss_recon_neg_voc = beta_voc*keras.losses.mean_squared_error(vocals,target_var2[:,3:4,:,:])

    train_loss_recon_bass = keras.losses.mean_squared_error(bass,target_var2[:,1:2,:,:])
    alpha_component += alpha*keras.losses.mean_squared_error(bass,target_var2[:,0:1,:,:])
    alpha_component += alpha*keras.losses.mean_squared_error(bass,target_var2[:,2:3,:,:])
    train_loss_recon_neg = beta*keras.losses.mean_squared_error(bass,target_var2[:,3:4,:,:])

    train_loss_recon_drums = keras.losses.mean_squared_error(drums,target_var2[:,2:3,:,:])
    alpha_component += alpha*keras.losses.mean_squared_error(drums,target_var2[:,0:1,:,:])
    alpha_component += alpha*keras.losses.mean_squared_error(drums,target_var2[:,1:2,:,:])
    train_loss_recon_neg += beta*keras.losses.mean_squared_error(drums,target_var2[:,3:4,:,:])

    vocals_error=train_loss_recon_vocals.sum()
    drums_error=train_loss_recon_drums.sum()
    bass_error=train_loss_recon_bass.sum()
    negative_error=train_loss_recon_neg.sum()
    negative_error_voc=train_loss_recon_neg_voc.sum()
    alpha_component=alpha_component.sum()

    loss=abs(vocals_error+drums_error+bass_error-negative_error-alpha_component-negative_error_voc)

    
    # weights=model.get_weights()
    params1 = lasagne.layers.get_all_params(network2, trainable=True)

    #A partir de aquí se haría al compilar/entrenar el modelo en Keras CONSULTAR SI HACE FALTA BAJAR A TF PARA HACER SESSION.RUN
    #optimizer=adadelta
    updates = lasagne.updates.adadelta(loss, params1)

    # val_updates=lasagne.updates.nesterov_momentum(loss1, params1, learning_rate=0.00001, momentum=0.7)

    train_fn = theano.function([input_var2,target_var2], loss, updates=updates,allow_input_downcast=True)

    train_fn1 = theano.function([input_var2,target_var2], [vocals_error,bass_error,drums_error,negative_error,alpha_component,negative_error_voc], allow_input_downcast=True)

    predict_function2=theano.function([input_var2],[vocals,bass,drums,others],allow_input_downcast=True)

    losser=[]
    loss2=[]

    if not skip_train:

        logging.info("Training...")
        for epoch in range(num_epochs):

            train_err = 0
            train_batches = 0
            vocals_err=0
            drums_err=0
            bass_err=0
            negative_err=0
            alpha_component=0
            beta_voc=0
            start_time = time.time()
            for batch in range(train.iteration_size): 
                inputs, target = train()
                jump = inputs.shape[2]
                inputs=np.reshape(inputs,(inputs.shape[0],1,inputs.shape[1],inputs.shape[2]))
                targets=np.ndarray(shape=(inputs.shape[0],4,inputs.shape[2],inputs.shape[3]))
                #import pdb;pdb.set_trace()
                targets[:,0,:,:]=target[:,:,:jump]
                targets[:,1,:,:]=target[:,:,jump:jump*2]
                targets[:,2,:,:]=target[:,:,jump*2:jump*3]
                targets[:,3,:,:]=target[:,:,jump*3:jump*4]
                target = None

                train_err+=train_fn(inputs,targets)
                [vocals_erre,bass_erre,drums_erre,negative_erre,alpha,betae_voc]=train_fn1(inputs,targets)
                vocals_err +=vocals_erre
                bass_err +=bass_erre
                drums_err +=drums_erre
                negative_err +=negative_erre
                beta_voc+=betae_voc
                alpha_component+=alpha
                train_batches += 1
        
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err/train_batches))
            losser.append(train_err / train_batches)
            print("  training loss for vocals:\t\t{:.6f}".format(vocals_err/train_batches))
            print("  training loss for bass:\t\t{:.6f}".format(bass_err/train_batches))
            print("  training loss for drums:\t\t{:.6f}".format(drums_err/train_batches))
            print("  Beta component:\t\t{:.6f}".format(negative_err/train_batches))
            print("  Beta component for voice:\t\t{:.6f}".format(beta_voc/train_batches))
            print("  alpha component:\t\t{:.6f}".format(alpha_component/train_batches))
            losser.append(train_err / train_batches)
            save_model(model,network2)

    if not skip_sep:

        logging.info("Separating")
        source = ['vocals','bass','drums','other']
        dev_directory = os.listdir(os.path.join(testdir,"Dev"))
        test_directory = os.listdir(os.path.join(testdir,"Test")) #we do not include the test dir
        dirlist = []
        dirlist.extend(dev_directory)
        dirlist.extend(test_directory)
        for f in sorted(dirlist):
            if not f.startswith('.'):
                if f in dev_directory:
                    song=os.path.join(testdir,"Dev",f,"mixture.wav")
                else:
                    song=os.path.join(testdir,"Test",f,"mixture.wav")
                audioObj, sampleRate, bitrate = util.readAudioScipy(song)
                
                assert sampleRate == 44100,"Sample rate needs to be 44100"

                audio = (audioObj[:,0] + audioObj[:,1])/2
                audioObj = None
                mag,ph=transform.compute_file(audio,phase=True)
         
                mag=scale_factor*mag.astype(np.float32)

                batches,nchunks = util.generate_overlapadd(mag,input_size=mag.shape[-1],time_context=train.time_context,overlap=train.overlap,batch_size=train.batch_size,sampleRate=sampleRate)
                output=[]

                batch_no=1
                for batch in batches:
                    batch_no+=1
                    start_time=time.time()
                    output.append(predict_function2(batch))

                output=np.array(output)
                mm=util.overlapadd_multi(output,batches,nchunks,overlap=train.overlap)

                #write audio files
                if f in dev_directory:
                    dirout=os.path.join(outdir,"Dev",f)
                else:
                    dirout=os.path.join(outdir,"Test",f)
                if not os.path.exists(dirout):
                    os.makedirs(dirout)
                for i in range(mm.shape[0]):
                    audio_out=transform.compute_inverse(mm[i,:len(ph)]/scale_factor,ph)
                    if len(audio_out)>len(audio):
                        audio_out=audio_out[:len(audio)]
                    util.writeAudioScipy(os.path.join(dirout,source[i]+'.wav'),audio_out,sampleRate,bitrate)
                    audio_out=None 
                audio = None

    return losser  


"""
# If you want initialize the model

from keras import backend as K
def reset_weights(model):
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
            
reset_weights(model)

"""