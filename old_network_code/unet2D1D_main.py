# ==================================================================================================
#
# ==================================================================================================
# --------------------------------------------------------------------------------------------------

import tensorflow as tf
import sys
import h5py
import numpy

import unet2D1D_load as load3D
import unet2D1D_arch as unet2D1D

# --------------------------------------------------------------------------------------------------
defaultMode               = 'training'
defaultInputDataPathRecon = 'matlab_3D_data_sets/train_3D_data_set.mat'
defaultOutputDataPath     = 'network/unet2D1D.ckpt'
defaultTrainIter          =  10000 # TODO: handle it differently and include satisfactory loss func.
defaultTemp               = "testData_Unet.h5"
defaultSavedNetworkPath   = 'temp_network/Unet2D1D.ckpt'

# --------------------------------------------------------------------------------------------------
def training_network\
(
    dataStruct,
    outputDataPath,
    net,
    bsize
):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("The maximum iteration is {}".format(defaultTrainIter))
        for i in range( defaultTrainIter ):
            batch = dataStruct.data.next_batch(bsize) #change it to be externally set
            net.train_step.run(feed_dict={net.imag: batch[0], net.true: batch[1]})
            train_accuracy = net.accuracy.eval(feed_dict={net.imag: batch[0], net.true: batch[1]})

            if i % 25 == 0:
                testPosit = net.test_pos.eval(feed_dict={
                    net.imag: batch[0], net.true: batch[1]})
                print('step %d, training accuracy %g, Positivity Constraint %g' % (i, train_accuracy, testPosit))

                save_path = net.saver.save(sess, outputDataPath)
                print("Model saved in file: %s" % save_path)

            if train_accuracy < 30:
                break

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
    net = unet2D1D.create( dataStruct.data.images.shape, batchSize )

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
    print("RUNNING DEFAULT PARAMETERS")
    main()
else:
    main( sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] )

