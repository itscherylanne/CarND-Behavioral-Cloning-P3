import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, MaxPooling2D
from keras.layers import Convolution2D, Cropping2D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import keras.backend as K
from keras.regularizers import l2, activity_l2

angular_offset = 0.2


#-----------------------------------
# Data Loading
#-----------------------------------
def ReadCSVFile(filename):
    lines = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            draw = np.random.uniform(0,1)
            #if(float(line[6])>0.5 and (float(line[3]) != 0 or (float(line[3])==0 and draw > 0.95))):
            if (float(line[3]) != 0 and float(line[6]) > 0.5 ):
                lines.append(line)
    return lines

def GetCurrentPath(source_path, path_to_img):
      filename = source_path.split('/')[-1]
      current_path = path_to_img + filename
      #print(current_path)
      return current_path


def OverwriteImagePaths(lines, img_path):
    for line in lines:
        line[0] = GetCurrentPath( line[0], img_path)
        line[1] = GetCurrentPath( line[1], img_path)
        line[2] = GetCurrentPath( line[2], img_path)
    return lines


#-----------------------------------
# Generator Related Functions
#-----------------------------------
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                angle_val = float(batch_sample[3])
                #----Center Image------
                center_name = batch_sample[0]
                temp_image = cv2.imread(center_name)
                center_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB)
                #center_image = ResizeImage(temp_image)
                center_angle = angle_val
                images.append(center_image)
                angles.append(center_angle)
                #----Left Image------
                left_name = batch_sample[0]
                temp_image = cv2.imread(left_name)
                left_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB)
                #left_image = ResizeImage(temp_image)
                left_angle = angle_val+angular_offset
                images.append(left_image)
                angles.append(left_angle)
                #----Right Image------
                right_name = batch_sample[0]
                temp_image = cv2.imread(right_name)
                right_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB)
                #right_image = ResizeImage(temp_image)
                right_angle = angle_val-angular_offset
                images.append(right_image)
                angles.append(right_angle)
                #----Flipped: Center Image------
                center_image_flipped = np.fliplr(center_image)
                images.append(center_image_flipped)
                angles.append(-center_angle)
                #----Flipped: Left Image------
                left_image_flipped = np.fliplr(left_image)
                images.append(left_image_flipped)
                angles.append(-left_angle)
                #----Flipped: Right Image------
                right_image_flipped = np.fliplr(right_image)
                images.append(right_image)
                angles.append(-right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

#-----------------------------------
# Main Function
#-----------------------------------
if __name__ == '__main__':

    print('---Clearing keras backend to avoid memory issue-----')

    config = K.tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = K.tf.Session(config=config)

    print('-------Track 1 Data---------')
    lines = []
    lines = ReadCSVFile('./data/driving_log.csv')
    lines = OverwriteImagePaths(lines, './data/IMG/')
    print('Number of entries from Track 1: ', len(lines))
    #print(lines[0][0])
    #print(lines[0][1])

    print('Number of entries: ', len(lines))
    print('-------Track 2 Data---------')
    lines2 = []
    #lines2 = ReadCSVFile('./run3/driving_log.csv')
    #lines2 = OverwriteImagePaths(lines2, './run3/IMG/')
    print('Number of entries from Track 2: ', len(lines2))
    lines3 = lines + lines2
    lines = []
    lines2 = []
    print('Number of combined entries: ', len(lines3))

    print('-------Splitting data---------')
    train_samples, validation_samples = train_test_split(lines3, test_size=0.2)
    #train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    print('Training Samples: ', len(train_samples))
    print('Validation Samples: ', len(validation_samples))

    print('-------Creating Generators---------')
    train_generator = generator(train_samples, batch_size=4)
    validation_generator = generator(validation_samples, batch_size=4)

    print('-------Creating Training Model---------')
    model = Sequential()
    model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (50,50)), input_shape=(160,320,3)))
    model.add(Convolution2D(24,5,5,activation='relu',subsample=(2, 2),border_mode='valid'))#, W_regularizer=l2(0.01), init='normal'))#, activity_regularizer=activity_l2(0.01)))
    model.add(Dropout(.1))
    model.add(Convolution2D(36,5,5,activation='relu',subsample=(2, 2),border_mode='valid'))#, W_regularizer=l2(0.01), init='normal'))#, activity_regularizer=activity_l2(0.01)))
    model.add(Dropout(.1))
    model.add(Convolution2D(48,5,5,activation='relu',subsample=(2, 2),border_mode='valid'))#, W_regularizer=l2(0.01), init='normal'))#, activity_regularizer=activity_l2(0.01)))
    model.add(Dropout(.1))
    model.add(Convolution2D(64,3,3,activation='relu',border_mode='valid'))#, W_regularizer=l2(0.01), init='normal'))#, activity_regularizer=activity_l2(0.01)))
    model.add(Dropout(.1))
    model.add(Convolution2D(64,3,3,activation='relu',border_mode='valid'))#, W_regularizer=l2(0.01), init='normal'))#, activity_regularizer=activity_l2(0.01)))
    model.add(Dropout(.1))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))#, W_regularizer=l2(0.01), init='normal'))#, activity_regularizer=activity_l2(0.01)))
    #model.add(Dropout(.1))
    model.add(Dense(50, activation='relu'))#, W_regularizer=l2(0.01), init='normal'))#, activity_regularizer=activity_l2(0.01)))
    #model.add(Dropout(.1))
    model.add(Dense(10, activation='relu'))#, W_regularizer=l2(0.01), init='normal'))#, activity_regularizer=activity_l2(0.01)))
    #model.add(Dropout(.1))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')



    print( model.summary() )

    print('-------Training Model with Generator---------')
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=0, mode='auto')
    model_checkpoint = ModelCheckpoint("model.h5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')

    #print('number of epochs: 10')
    #model.fit_generator(train_generator, nb_epoch=20, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), callbacks=[early_stopping, model_checkpoint])
    model.fit_generator(train_generator, nb_epoch=3, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples))


    model.save('model.h5')
    print('-------Model Saved. Exiting---------')
    exit()
