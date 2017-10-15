import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from train import *

import os

## converts arrayo to sparse svmlight format
def fromsparsetofile(filename, array, deli1=" ", deli2=":",ytarget=None):
    zsparse=csr_matrix(array)
    indptr = zsparse.indptr
    indices = zsparse.indices
    data = zsparse.data
    print(" data lenth %d" % (len(data)))
    print(" indices lenth %d" % (len(indices)))
    print(" indptr lenth %d" % (len(indptr)))

    f=open(filename,"w")
    counter_row=0
    for b in range(0,len(indptr)-1):
        #if there is a target, print it else , print nothing
        if (ytarget is not None):
             f.write(str(ytarget[b]) + deli1)

        for k in range(indptr[b],indptr[b+1]):
            if (k==indptr[b]):
                if np.isnan(data[k]):
                    f.write("%d%s%f" % (indices[k],deli2,-1))
                else :
                    f.write("%d%s%f" % (indices[k],deli2,data[k]))
            else :
                if np.isnan(data[k]):
                     f.write("%s%d%s%f" % (deli1,indices[k],deli2,-1))
                else :
                    f.write("%s%d%s%f" % (deli1,indices[k],deli2,data[k]))
        f.write("\n")
        counter_row+=1
        if counter_row%10000==0:
            print(" row : %d " % (counter_row))
    print('finish')
    f.close()

#creates the main dataset abd prints 2 files to dataset2_train.txt and  dataset2_test.txt

def stacknet_prepare_validate():
    # 2016 train, validation
    from features import feature_list_non_linear
    feature_list = feature_list_non_linear.feature_list_all
    prop2016 = prepare_features(2016, feature_list, True)
    transactions = utils.load_transaction_data(2016)
    # merge transaction and prop data
    df2016 = transactions.merge(prop2016, how='inner', on='parcelid')
    df2016['transaction_year'] = 0
    df2016['transaction_month'] = df2016['transactiondate'].map(lambda x: x.date().month)
    df2016['transaction_quarter'] = df2016['transaction_month'].map({1:1, 2:1, 3:1, 4:2, 5:2, 6:2, 7:3, 8:3, 9:3, 10:4, 11:4, 12:4})
    df2016.drop('transaction_month', axis=1, inplace=True)

    transactions = None
    prop2016 = None
    gc.collect()

    prop2017 = prepare_features(2017, feature_list, True)
    transactions = utils.load_transaction_data(2017)
    # merge transaction and prop data
    df2017 = transactions.merge(prop2017, how='inner', on='parcelid')
    df2017['transaction_year'] = 1
    df2017['transaction_month'] = df2017['transactiondate'].map(lambda x: x.date().month)
    df2017['transaction_quarter'] = df2017['transaction_month'].map({1:1, 2:1, 3:1, 4:2, 5:2, 6:2, 7:3, 8:3, 9:3, 10:4, 11:4, 12:4})
    df2017.drop('transaction_month', axis=1, inplace=True)

    transactions = None
    prop2017 = None
    gc.collect()

    df_all = pd.concat([df2016, df2017])

    df_train2016, df_validate2016, df_train_all, df_validate_all = get_train_validate_split(df2016, df_all)

    df_train2016 = df_train2016[(df_train2016.logerror > -0.4) & (df_train2016.logerror < 0.419)]
    df_train_all = df_train_all[(df_train_all.logerror > -0.4) & (df_train_all.logerror < 0.419)]

    # 2016
    # validation set need parcelid and transactiondate as unqiue identifier of rows
    df_train2016.drop(['parcelid', 'transactiondate'], axis=1, inplace=True)
    X_validate2016_id_date = df_validate2016[['parcelid', 'transactiondate']]
    df_validate2016.drop(['parcelid', 'transactiondate'], axis=1, inplace=True)
    X_train2016, y_train2016 = utils.get_features_target(df_train2016)
    X_validate2016, _ = utils.get_features_target(df_validate2016)
    print('train 2016')
    print(X_train2016.shape, y_train2016.shape)
    print('validate 2016')
    print(X_validate2016.shape)
    # for col in X_train2016:
    #     print(col, X_train2016[col].dtype)

    validate_folder = 'data/stacknet/validation'
    if not os.path.exists(validate_folder):
        os.makedirs(validate_folder)
    X_validate2016_id_date.to_csv("%s/validate2016_id_date.csv" %validate_folder, index=False)
    X_train2016 = X_train2016.values.astype(np.float32, copy=False)
    X_validate2016 = X_validate2016.values.astype(np.float32, copy=False)
    y_train2016 = y_train2016.values.astype(np.float32, copy=False)
    fromsparsetofile("%s/train2016.txt" %validate_folder, X_train2016, deli1=" ", deli2=":",ytarget=y_train2016)
    fromsparsetofile("%s/validate2016.txt" %validate_folder, X_validate2016, deli1=" ", deli2=":",ytarget=None)
    print (" finished with 2016 train validation data" )

    # 2016 - 2017
    # validation set need parcelid and transactiondate as unqiue identifier of rows
    df_train_all.drop(['parcelid', 'transactiondate'], axis=1, inplace=True)
    X_validate_all_id_date = df_validate_all[['parcelid', 'transactiondate']]
    df_validate_all.drop(['parcelid', 'transactiondate'], axis=1, inplace=True)
    X_train_all, y_train_all = utils.get_features_target(df_train_all)
    X_validate_all, _ = utils.get_features_target(df_validate_all)
    print('train all')
    print(X_train_all.shape, y_train_all.shape)
    print('validate all')
    print(X_validate_all.shape)

    X_validate_all_id_date.to_csv("%s/validate_all_id_date.csv" %validate_folder, index=False)
    X_train_all = X_train_all.values.astype(np.float32, copy=False)
    X_validate_all = X_validate_all.values.astype(np.float32, copy=False)
    y_train_all = y_train_all.values.astype(np.float32, copy=False)
    fromsparsetofile("%s/train_all.txt" %validate_folder, X_train_all, deli1=" ", deli2=":",ytarget=y_train_all)
    fromsparsetofile("%s/validate_all.txt" %validate_folder, X_validate_all, deli1=" ", deli2=":",ytarget=None)
    print (" finished with all train validation data" )


def stacknet_prepare_test2017():
    # 2016 train, validation
    from features import feature_list_non_linear
    feature_list = feature_list_non_linear.feature_list_all
    prop2016 = prepare_features(2016, feature_list, True)
    transactions = utils.load_transaction_data(2016)
    # merge transaction and prop data
    df2016 = transactions.merge(prop2016, how='inner', on='parcelid')
    df2016['transaction_year'] = 0
    df2016['transaction_month'] = df2016['transactiondate'].map(lambda x: x.date().month)
    df2016['transaction_quarter'] = df2016['transaction_month'].map({1:1, 2:1, 3:1, 4:2, 5:2, 6:2, 7:3, 8:3, 9:3, 10:4, 11:4, 12:4})
    df2016.drop('transaction_month', axis=1, inplace=True)

    del prop2016; del transactions
    gc.collect()

    prop2017 = prepare_features(2017, feature_list, True)
    transactions = utils.load_transaction_data(2017)
    # merge transaction and prop data
    df2017 = transactions.merge(prop2017, how='inner', on='parcelid')
    df2017['transaction_year'] = 1
    df2017['transaction_month'] = df2017['transactiondate'].map(lambda x: x.date().month)
    df2017['transaction_quarter'] = df2017['transaction_month'].map({1:1, 2:1, 3:1, 4:2, 5:2, 6:2, 7:3, 8:3, 9:3, 10:4, 11:4, 12:4})
    df2017.drop('transaction_month', axis=1, inplace=True)

    df_all = pd.concat([df2016, df2017])
    del df2016; del df2017; del transactions;
    gc.collect()

    df_all = df_all[(df_all.logerror > -0.4) & (df_all.logerror < 0.419)]

    # 2016 - 2017
    # validation set need parcelid and transactiondate as unqiue identifier of rows
    df_all.drop(['parcelid', 'transactiondate'], axis=1, inplace=True)
    prop2017_id = prop2017['parcelid']
    prop2017.drop('parcelid', axis=1, inplace=True)
    X_train_all, y_train_all = utils.get_features_target(df_all)
    print('train all')
    print(X_train_all.shape, y_train_all.shape)
    print('test 2017')
    print(prop2017.shape)

    prop2017_id.to_csv("%s/test2017_id.csv" %test_folder, index=False)
    X_train_all = X_train_all.values.astype(np.float32, copy=False)
    y_train_all = y_train_all.values.astype(np.float32, copy=False)
    print('2017 train shape')
    print(X_train_all.shape)
    fromsparsetofile("%s/train2017_test.txt" %test_folder, X_train_all, deli1=" ", deli2=":",ytarget=y_train_all)

    # year = 2017
    # for month in [10, 11, 12]:
    #     prop2017['transactiondate'] = '%s-%s-30' % (year, month)
    #     prop2017 = utils.add_date_features(prop2017)
    #     prop2017.drop(['transactiondate'], inplace=True, axis=1)
    #     prop2017 = prop2017.values.astype(np.float32, copy=False)
    #     print('%d 2017 prop shape' %month)
    #     print(prop2017.shape)
    #     fromsparsetofile("%s/test2017%d.txt" %(test_folder, month), prop2017, deli1=" ", deli2=":",ytarget=None)

    prop2017['transaction_year'] = 1
    prop2017['transaction_quarter'] = 4
    prop2017 = prop2017.values.astype(np.float32, copy=False)
    print('2017 prop shape')
    print(prop2017.shape)
    fromsparsetofile("%s/test2017%d.txt" %(test_folder, month), prop2017, deli1=" ", deli2=":",ytarget=None)

    print (" finished with 2017 train test data" )

def stacknet_prepare_test2016():
    # 2016 train, validation
    from features import feature_list_non_linear
    feature_list = feature_list_non_linear.feature_list_all
    prop2016 = prepare_features(2016, feature_list, True)
    transactions = utils.load_transaction_data(2016)
    # merge transaction and prop data
    df2016 = transactions.merge(prop2016, how='inner', on='parcelid')
    df2016['transaction_year'] = 0
    df2016['transaction_month'] = df2016['transactiondate'].map(lambda x: x.date().month)
    df2016['transaction_quarter'] = df2016['transaction_month'].map({1:1, 2:1, 3:1, 4:2, 5:2, 6:2, 7:3, 8:3, 9:3, 10:4, 11:4, 12:4})
    df2016.drop('transaction_month', axis=1, inplace=True)

    transactions = None
    gc.collect()

    df2016 = df2016[(df2016.logerror > -0.4) & (df2016.logerror < 0.419)]

    # 2016
    # validation set need parcelid and transactiondate as unqiue identifier of rows
    df2016.drop(['parcelid', 'transactiondate'], axis=1, inplace=True)
    prop2016_id = prop2016['parcelid']
    prop2016.drop('parcelid', axis=1, inplace=True)
    X_train2016, y_train2016 = utils.get_features_target(df2016)
    print('train 2016')
    print(X_train2016.shape, y_train2016.shape)
    print('test 2016')
    print(prop2016.shape)
    # for col in X_train2016:
    #     print(col, X_train2016[col].dtype)

    test_folder = 'data/stacknet/test'
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    prop2016_id.to_csv("%s/test2016_id.csv" %test_folder, index=False)
    X_train2016 = X_train2016.values.astype(np.float32, copy=False)
    y_train2016 = y_train2016.values.astype(np.float32, copy=False)
    print('2016 train shape')
    print(X_train2016.shape)
    fromsparsetofile("%s/train2016_test.txt" %test_folder, X_train2016, deli1=" ", deli2=":",ytarget=y_train2016)

    # year = 2016
    # for month in [10, 11, 12]:
    #     prop2016['transactiondate'] = '%s-%s-30' % (year, month)
    #     prop2016 = utils.add_date_features(prop2016)
    #     prop2016.drop(['transactiondate'], inplace=True, axis=1)
    #     prop2016 = prop2016.values.astype(np.float32, copy=False)
    #     print('%d 2017 prop shape' %month)
    #     print(prop2016.shape)
    #     fromsparsetofile("%s/test2016%d.txt" %(test_folder, month), prop2016, deli1=" ", deli2=":",ytarget=None)

    print('2016 prop data')
    prop2016['transaction_year'] = 0
    prop2016['transaction_quarter'] = 4
    print('finish adding time')
    for col in prop2016.columns:
        if prop2016[col].dtype != 'float32':
            prop2016[col] = prop2016[col].astype('float32')
    print('2016 prop shape')
    print(prop2016.shape)
    fromsparsetofile("%s/test2016.txt" %(test_folder), prop2016, deli1=" ", deli2=":",ytarget=None)

    print (" finished with 2016 train test data" )

if __name__ == '__main__':
    # stacknet_prepare_validate()
    stacknet_prepare_test2016()
