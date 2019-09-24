import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error
import xgboost as xgb

test = pd.read_csv('testA.csv',header=None)
rice = pd.read_csv('train_rice.csv', encoding='gbk')
weather = pd.read_csv('train_weather.csv', encoding='gbk')
weather.loc[(weather['日期']<=15),'时间窗'] = weather.loc[(weather['日期']<=15),'月份'].apply(lambda x: str(x)+'_1')
weather.loc[(weather['日期']>15),'时间窗'] = weather.loc[(weather['日期']>15),'月份'].apply(lambda x: str(x)+'_2')
#weather.loc[(weather['日期']<=7),'时间窗'] = weather.loc[(weather['日期']<=7),'月份'].apply(lambda x: str(x)+'_1')
#weather.loc[weather['日期'].between(7,14),'时间窗'] = weather.loc[weather['日期'].between(7,14),'月份'].apply(lambda x: str(x)+'_2')
#weather.loc[(weather['日期']>14),'时间窗'] = weather.loc[(weather['日期']>14),'月份'].apply(lambda x: str(x)+'_3')
rice_na = rice.iloc[:,1:].isnull()
col_wind = ['02时风向', '08时风向', '14时风向', '20时风向']
for col in col_wind:
    weather[col] = weather[col].replace('/',np.NaN)
    weather[col] = weather[col].fillna(method='ffill')
    weather[col] = weather[col].astype('category')

col2float = ['日照时数（单位：h)','日平均风速(单位：m/s)','日降水量（mm）',
            '日平均温度（单位：℃）', '日相对湿度（单位：%）', '日平均气压（单位：hPa）']

def o2float(x) :
    try :
        t = float(x)
    except :
        t = np.nan
    return t

for col in col2float:
    weather[col] = weather[col].apply(o2float)

for col in weather.columns:
    if weather[col].dtype.name == 'float64':
        weather[col].fillna(weather[col].mean(), inplace=True)

def get_features(origin_data,tpe='早稻',datasets='train'):
    
    #筛选站名
    origin_data = origin_data.query('站名id==1')
    origin_data = origin_data.drop('站名id',axis=1)
    #修改列名
    origin_data = origin_data.rename(columns=lambda x: x.replace('(','（').split('（')[0])
    #获取参数
    if tpe == '早稻':
        month_begin = 3
        month_end = 8
    else:
        month_begin = 5
        month_end = 11
        
    if datasets == 'train':
        year = [2015,2016,2017]
    else:
        year = [2018]
        
    #筛选数据
    df = origin_data[(origin_data['月份']>=month_begin)&(origin_data['月份']<=month_end)]
    df = df[df['年份'].isin(year)]
    
    #构造特征
#     numerical_feature_cols = df.columns[np.r_[4,9:16]]
    numerical_feature_cols = ['日照时数', '日平均风速', '日降水量', '日最高温度', '日最低温度', '日平均温度', '日相对湿度', '日平均气压']
#     categorical_feature_cols = df.columns[5:9]
    categorical_feature_cols = ['02时风向', '08时风向', '14时风向', '20时风向']
    
    numerical_data = pd.DataFrame()
    for col in numerical_feature_cols:
        df_temp = df.groupby(['区县id','年份','月份','时间窗'])[col].agg({f'{col}_mean':'mean',f'{col}_max':'max',f'{col}_min':'min',f'{col}_std':'std'})
        #df_temp = df.groupby(['区县id','年份','半月'])[col].agg({f'{col}_mean':'mean',f'{col}_max':'max',f'{col}_min':'min',f'{col}_std':'std'})
        #df_temp = df.groupby(['区县id','年份','月份'])[col].agg({f'{col}_mean':'mean',f'{col}_max':'max',f'{col}_min':'min',f'{col}_std':'std'})
        numerical_data = pd.concat([df_temp,numerical_data],axis=1)
    
    #风向分类统计数量
    categorical_data_list = []
    for col in categorical_feature_cols:
        #series_temp = df.groupby(['区县id','年份','月份'])[col].value_counts()
        series_temp = df.groupby(['区县id','年份','月份','时间窗'])[col].value_counts()
        categorical_data_list.append(series_temp)
    categorical_data = pd.concat(categorical_data_list,axis=1)
    #fillna
    categorical_data = categorical_data.fillna(0).astype(int)
    #标记不同时段
    categorical_data = categorical_data.unstack()
    first_levels = categorical_data.columns.levels[0]
    categorical_data_list = []
    for first_level in first_levels:
        categorical_data_list.append(categorical_data[first_level].rename(columns=lambda x: f'{first_level}_{x}'))
    categorical_data = pd.concat(categorical_data_list,axis=1).fillna(0).astype(int)
    
    #合并features
    feature_data = pd.concat([numerical_data,categorical_data],axis=1)
    
    #按照月份展开
    feature_data = feature_data.unstack()
    first_level = feature_data.columns.levels[0]
    
    feature_data_list = []
    for level in first_level:
        feature_data_list.append(feature_data[level].rename(columns=lambda x: f'{x}月_{level}'))
    feature_data = pd.concat(feature_data_list,axis=1).fillna(0).astype(int).reset_index()
    
    return feature_data

def get_label(train_rice,tpe='早稻'):

    #stack
    temp = pd.DataFrame(train_rice.set_index('区县id').stack()).reset_index().rename(columns={0:'产量'})
    #reset_index
    label = temp.join(temp['level_1'].str.split('年',expand=True).rename(columns={0:'年份',1:'水稻种类'})).drop('level_1',axis=1)
    label['年份'] = label['年份'].astype(int)
    #筛选
    label = label[label['水稻种类']==tpe].drop('水稻种类',axis=1)
    
    return label

def get_train_and_test_data(train_weather,train_rice,tpe='早稻'):

    feature_data_train = get_features(train_weather,tpe)    
    label = get_label(train_rice,tpe)    
    train_data = feature_data_train.merge(label,on=['区县id','年份'])
    train_data['区县id'] = train_data['区县id'].str.replace('county','').astype('category')
    #drop掉无用列
#     train_data = train_data.drop(['区县id','年份'],axis=1)
    train_data = train_data.drop(['年份'],axis=1)
    
    test_data = get_features(train_weather,tpe,datasets='test')
    test_data['区县id'] = test_data['区县id'].str.replace('county','').astype('category')
    test_data = test_data.drop('年份',axis=1)
    return (train_data,test_data)        

def readdata(tpe='早稻'):
    train_data,test_data = get_train_and_test_data(weather, rice, tpe)
    
    X = train_data.iloc[:,:-1].values
    y = train_data.iloc[:,-1].values
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    test_X = test_data.iloc[:,:].values
    test_scaler = MinMaxScaler()
    test_X = test_scaler.fit_transform(test_X)
    return X,y,test_X,test_data

def train(tpe='早稻'):
    
    #f_w = []
    #for c in train_data.columns[1:]:
    #    if '区县id' in c:
    #        f_w.append(10)
    #    elif '温度' in c or '湿度' in c:
    #        f_w.append(8)
    #    elif '日照' in c:
    #        f_w.append(10)
    #    elif '降雨量' in c:
    #        f_w.append(8)
    #    else:
    #        f_w.append(5)
    
    X,y,test_X,test_data = readdata(tpe)
    
    print("Parameter optimization")
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
    clf = GridSearchCV(xgb_model,
                       #{'max_depth': [5,6,7,8,9,10],
                       {'max_depth': [7],
                        'n_estimators': [1300]}, verbose=2 ,n_jobs=44, cv=10)
    #clf.fit(X,y,sample_weight=features_w)
    clf.fit(X,y)
    print('best score: %s'%clf.best_score_)
    print('best params: %s'%clf.best_params_)
    
    model = clf.best_estimator_
    model.save_model('%s.model'%tpe)
    predictions = model.predict(X)
    actuals = y
    print('mse: %s'%(mean_squared_error(actuals, predictions)/(2*81)))
    
    return model, test_X, test_data
    
def predict(model,test_X, test_data, tpe='早稻'):
    test_y = model.predict(test_X).astype('float64')
    print(test_y[0])
    result = pd.concat([test_data['区县id'], pd.Series(test_y)],axis=1)
    result['区县id'] = result['区县id'].astype('int64')
    result = result.sort_values('区县id')
    result['区县id'] = result['区县id'].apply(lambda x: 'county'+str(x))
    result = result.rename(columns={0:'2018年%s'%tpe})
    rice_new = rice.merge(result, on=['区县id'])
    print('diff')
    print('16-15')
    print(np.mean(rice_new['2016年%s'%tpe]- rice_new['2015年%s'%tpe]))
    print('17-16')
    print(np.mean(rice_new['2017年%s'%tpe]- rice_new['2016年%s'%tpe]))
    print('18-17')
    print(np.mean(rice_new['2018年%s'%tpe]- rice_new['2017年%s'%tpe]))
    result = test.merge(result,left_on=[0],right_on=['区县id']).iloc[:,1:]
    result = result.groupby('区县id').agg({'2018年%s'%tpe: np.mean})
    result[0] = result.index
    result = test.merge(result,left_on=[0],right_on=[0])
    result.to_csv('%s.csv'%tpe,index=False,header=False)

def loadmodel(name='早稻'):
    model = xgb.XGBRegressor()
    model.load_model('%s.model'%name)
    return model

def run(tpe='早稻'):
    model, test_X, test_data = train(tpe)
    predict(model, test_X, test_data, tpe)

#run()
#run('晚稻')
