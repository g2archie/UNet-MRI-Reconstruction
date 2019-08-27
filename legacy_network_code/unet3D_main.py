
import tensorflow as tf
import sys
import h5py
import numpy

import unet3D_load as load3D
import unet3D_arch as unet3D

from utils import send_email

# --------------------------------------------------------------------------------------------------
defaultMode               = 'training'
defaultInputDataPathRecon = 'matlab_3D_data_sets/train_3D_data_set.mat'
defaultOutputDataPath     = 'network/unet2D1D.ckpt'
defaultTrainIter          =  5000 # TODO: handle it differently and include satisfactory loss func.
defaultTemp               = "testData_Unet.h5"
defaultSavedNetworkPath   = 'temp_network/Unet2D1D.ckpt'

def training_network\
(
    dataStruct,
    outputDataPath,
    net,
    bsize
):
    with tf.Session() as sess:
        accuracy_list = []
        early_stopping = False
        sess.run(tf.global_variables_initializer())
        print("The maximum iteration is {}".format(defaultTrainIter))
        try:
            send_email("The training has started", "Number of iterations is {}, model stored path is {}".format(defaultTrainIter, outputDataPath))
        except:
            print("sending has failed")
        #for i in range( defaultTrainIter ):
        i = -1
        for i in range( defaultTrainIter ):
            batch = dataStruct.data.next_batch(bsize) #change it to be externally set
            #print("input in batch has shape of {}, and true has shape of {}".format(batch[0].shape, batch[1].shape))
            net.train_step.run(feed_dict={net.imag: batch[0], net.true: batch[1]})
            i = i + 1
            if i % 5 == 0:
                train_accuracy = net.accuracy.eval(feed_dict={
                    net.imag: batch[0], net.true: batch[1]})
                testPosit = net.test_pos.eval(feed_dict={
                    net.imag: batch[0], net.true: batch[1]})
                added_loss = net.added_loss.eval(feed_dict={
                    net.imag: batch[0], net.true: batch[1]})
                loss = net.loss.eval(feed_dict={
                    net.imag: batch[0], net.true: batch[1]})
                print('step {}, training accuracy {}, Positivity Constraint {}, added_loss {}, shape of added_loss is {}'.format(i, train_accuracy, testPosit, added_loss, added_loss.shape))
                print("The shape of loss is {}, and its value is {}, shape of accuracy is {}".format(loss.shape, loss, train_accuracy.shape))
                accuracy_list.append(train_accuracy)
                save_path = net.saver.save(sess, outputDataPath)
                print("Model saved in file: %s" % save_path)
                
                
                #if train_accuracy < 100 or (len(accuracy_list) > 9 and numpy.std(accuracy_list[-10:]) < 75):
                #    print("Early stopping triggered, accuracy is {}".format(train_accuracy))
                #    print("Last 10 training accuracy are {}".format(accuracy_list[-10:]))
                #    early_stopping = True
                #    finished_step = i
                #    break

        if early_stopping:
            body_text = "Training finished at {}, Early stopping triggered, accuracy is {}, Last 10 training accuracy are {}".format(finished_step, train_accuracy, accuracy_list[-10:])
        else:
            body_text = "Training finished, Last 10 training accuracy are {}".format(accuracy_list[-10:])
        try:    
            send_email("The training has ended", body_text)
        except:
            print("sending has failed")
        save_path = net.saver.save( sess, outputDataPath )
        print("Model saved in file: %s" % save_path)

        print('--------------------> DONE <--------------------')

        return

# --------------------------------------------------------------------------------------------------
def testing_network\
(
    dataStruct,
    savedNetworkPath,
    outputDataPath,
    net
):
    bsize = 8
    with tf.Session() as sess:
        sess.run( tf.global_variables_initializer() )
        net.saver.restore( sess, savedNetworkPath )
        print( "Model restored." )
        size = len(dataStruct.data.images) // bsize
        output = None
        for i in range(0, size):
          tmp = sess.run\
          (
              net.y_conv,
              feed_dict =
              {
                  #net.imag: dataStruct.data.images[0:bsize],
                  #net.true: dataStruct.data.true[0:bsize]
                  net.imag: dataStruct.data.images[i*bsize:(i+1)*bsize],
                  net.true: dataStruct.data.true[i*bsize:(i+1)*bsize]
              }
          )
          if output is None:
            output = tmp
          else:
            output = numpy.append(output, tmp, axis=0)
        
        fileOutName = outputDataPath
        fData = h5py.File(fileOutName, 'w')
        fData['result'] = numpy.array ( output                          )
        #fData['truth']  = numpy.array ( dataStruct.data.true[0:bsize]   )
        #fData['imag']   = numpy.array ( dataStruct.data.images[0:bsize] )
        fData['truth']  = numpy.array ( dataStruct.data.true[0:size*bsize]   )
        fData['imag']   = numpy.array ( dataStruct.data.images[0:size*bsize] )
        fData.close()

        print('--------------------> DONE <--------------------')

        return

# --------------------------------------------------------------------------------------------------
def main\
(
    trainOrTest = defaultMode,
    inputDataPathRecon = defaultInputDataPathRecon,
    inputDataPathTrue = defaultInputDataPathRecon,
    outputDataPath = defaultOutputDataPath,
    savedNetworkPath = defaultSavedNetworkPath
):

    batchSize = 8
    dataStruct = load3D.read_data_sets( inputDataPathRecon, inputDataPathTrue )
    net = unet3D.create( dataStruct.data.images.shape, batchSize )

    if trainOrTest == 'training':
        training_network( dataStruct, outputDataPath, net, batchSize )
    elif trainOrTest == 'test':
        testing_network( dataStruct, savedNetworkPath, outputDataPath, net )
    else:
        print("[ERROR] Unknown mode type.")
        raise

# --------------------------------------------------------------------------------------------------
# If not all arguments are provided run the default ones.
# NOTE: this is placeholder - proper handling of flags should be used.
expectedNumOfArguments = 6
if len(sys.argv) < expectedNumOfArguments:
    main()
else:
    main( sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] )

