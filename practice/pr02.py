import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

'''
'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
       'Fare', 'Cabin', 'Embarked'
'''

def titanic():
    sns.set_style('whitegrid')
    plt.figure(figsize=(20, 10))
    df = pd.read_csv('../data/train.csv', index_col='PassengerId')
    # print(df)
    print(df.columns)

    mask = df.Fare < 1000

    # 가족 유무에 대한 컬럼
    df = df[mask]  # 조건에 충족하는 값만 출력된다.
    df['fm'] = df.SibSp + df.Parch

    # 나이별로 구분
    # 결측치 0 혹은 평균 값
    # 평균 값 = df['Age'].mean()
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['a0'] = False
    df['a1'] = False
    df['a2'] = False
    df['a3'] = False
    df['a4'] = False
    df['a5'] = False
    df['a6'] = False
    df['a7'] = False
    df['a8'] = False


    def func(data):
        print(data.Age, data.a0, data.a1)
        if data.Age >= 0.0 and data.Age < 10.0:
            data.a0 = True
            return data
        elif data.Age >= 10.0 and data.Age < 20.0:
            data.a1 = True
            return data
        elif data.Age >= 20.0 and data.Age < 30.0:
            data.a2 = True
            return data
        elif data.Age >= 30.0 and data.Age < 40.0:
            data.a3 = True
            return data
        elif data.Age >= 40.0 and data.Age < 50.0:
            data.a4 = True
            return data
        elif data.Age >= 50.0 and data.Age < 60.0:
            data.a5 = True
            return data
        elif data.Age >= 60.0 and data.Age < 70.0:
            data.a6 = True
            return data
        elif data.Age >= 70.0 and data.Age < 80.0:
            data.a7 = True
            return data
        elif data.Age >= 80.0 and data.Age < 90.0:
            data.a8 = True
            return data
        return data




    df[['Age', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8']] = df[['Age', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8']].apply(func, axis=1)

    def func(data):
        if data >= 0.0 and data < 10.0:
            return 0
        elif data >= 10.0 and data < 20.0:
            return 10
        elif data >= 20.0 and data < 30.0:
            return 20
        elif data >= 30.0 and data < 40.0:
            return 30
        elif data >= 40.0 and data < 50.0:
            return 40
        elif data >= 50.0 and data < 60.0:
            return 50
        elif data >= 60.0 and data < 70.0:
            return 60
        elif data >= 70.0 and data < 80.0:
            return 70
        elif data >= 80.0 and data < 90.0:
            return 80



    # 나이대 구분
    df['age_group'] = df['Age'].apply(func)

#    df.to_csv('./sample.csv')
    # 기항지에 따른 티켓 값 분류
    for name, group in df.groupby('Embarked'):
        #  list(df.groupby('Embarked'))
        # 키 값
        print(name)
        # 실제 데이터
        # print(group)
    # C, Q, S
    df.loc[df['Embarked'] == 'C', 'Embarked'] = 0.0
    df.loc[df['Embarked'] == 'Q', 'Embarked'] = 1.0
    df.loc[df['Embarked'] == 'S', 'Embarked'] = 2.0
    df.loc[df['Embarked'] == None, 'Embarked'] = 3.0

    # 구간별, 나이별 평균 요금
    print(df.groupby(['Sex', 'age_group'])[['Fare']].mean())

    # df.to_csv('./sample.csv')

    # print(df.loc[df['fm'] != 0])

    # print(df)
    # plt.plot(df.Sex, df['fm'], marker='*', markerfacecolor='r', markersize=5)

    # 가족 유무에 따른 생존여부
    # sns.countplot(data=df, x='fm', hue='Survived')
    # 가족 유무에 다른 티켓 값
    # sns.barplot(x='fm', hue='Survived', y='Fare', data=df)
    # 성별에 따른 생존
    # sns.countplot( x='Sex', hue='Survived',  data=df )  # 판다스 데이터 프레임을 파라미터로 하여 x축컬럼이름, y축 컬럼이름
    # 성별에 따른 생존
    # sns.countplot( x='Sex', hue='Survived',  data=df )  # 판다스 데이터 프레임을 파라미터로 하여 x축컬럼이름, y축 컬럼이름
    # 기항지에 따른 요금및 생존
    # sns.barplot(x="Embarked", y="Fare", hue="Survived", data=df)
    # sns.countplot(x="Embarked", hue="Survived", data=df)
    # 나이대에 따른 생존
    sns.countplot(x="age_group",  hue="Survived", data=df)

    # sns.lineplot(data=df, y='Embarked', x='Fare', hue='Survived')

    # print(df['Age'].groupby(df['Age']).count())
    # sns.lineplot(x='Age', y=df.groupby(df['Age']).count(), data=df, palette="tab10", linewidth=2.5)

    # 정규분포
    # 빈도수에 로그를 입히면 최소값과 최대값의 차이 작아지면서 가운데로 온다 -> 수학을 활용한 특성공학

    df['with_f'] = df['fm'] == 1

    # print(df)
    perish = df[df['Survived'] == 0]
    survived = df[df['Survived'] == 1]

    # sns.distplot(perish['Fare'], hist=False)
    # sns.distplot(survived['Fare'], hist=False)


    y_train = df['Survived']
    y_train.to_csv('./y_train.csv', index=False)
    '''
    plt.show()
    # 로그
    import numpy as np
    df['LogFare'] = np.log(df['Fare'])
    plt.figure(figsize=(10, 8))
    sns.distplot(df['LogFare'], hist=False)
    sns.distplot(df['Fare'], hist=False)
    '''

def over_data():
    df = pd.read_csv('../data/train.csv', index_col='PassengerId')
    # print(df)


    mask = df.Fare < 1000

    # 가족 유무에 대한 컬럼
    df = df[mask]  # 조건에 충족하는 값만 출력된다.
    # 총 인원
    df['fm'] = df.SibSp + df.Parch + 1
    # 나이별로 구분
    # 결측치 0 혹은 평균 값
    # 평균 값 = df['Age'].mean()
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    def func(data):
        print(data.Age, data.a0, data.a1)
        if data.Age >= 0.0 and data.Age < 10.0:
            data.a0 = True
            return data
        elif data.Age >= 10.0 and data.Age < 20.0:
            data.a1 = True
            return data
        elif data.Age >= 20.0 and data.Age < 30.0:
            data.a2 = True
            return data
        elif data.Age >= 30.0 and data.Age < 40.0:
            data.a3 = True
            return data
        elif data.Age >= 40.0 and data.Age < 50.0:
            data.a4 = True
            return data
        elif data.Age >= 50.0 and data.Age < 60.0:
            data.a5 = True
            return data
        elif data.Age >= 60.0 and data.Age < 70.0:
            data.a6 = True
            return data
        elif data.Age >= 70.0 and data.Age < 80.0:
            data.a7 = True
            return data
        elif data.Age >= 80.0 and data.Age < 90.0:
            data.a8 = True
            return data
        return data




    def func(data):
        if data >= 0.0 and data < 10.0:
            return 0
        elif data >= 10.0 and data < 20.0:
            return 10
        elif data >= 20.0 and data < 30.0:
            return 20
        elif data >= 30.0 and data < 40.0:
            return 30
        elif data >= 40.0 and data < 50.0:
            return 40
        elif data >= 50.0 and data < 60.0:
            return 50
        elif data >= 60.0 and data < 70.0:
            return 60
        elif data >= 70.0 and data < 80.0:
            return 70
        elif data >= 80.0 and data < 90.0:
            return 80



    # 나이대 구분
    # df['age_group'] = df['Age'].apply(func)


    # 기항지에 따른 티켓 값 분류
    for name, group in df.groupby('Embarked'):
        #  list(df.groupby('Embarked'))
        # 키 값
        print(name)
        # 실제 데이터
        # print(group)
    # C, Q, S
    df.loc[df['Embarked'] == 'C', 'Embarked'] = 0.0
    df.loc[df['Embarked'] == 'Q', 'Embarked'] = 1.0
    df.loc[df['Embarked'] == 'S', 'Embarked'] = 2.0
    df.loc[df['Embarked'].isnull(), 'Embarked'] = 3.0

    # 성별
    df.loc[df['Sex'] == 'male', 'Sex'] = 0.0
    df.loc[df['Sex'] == 'female', 'Sex'] = 1.0

    '''
    '', 'Pclass', '', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
           'Fare', '', 'Embarked'
    '''

    df = df.drop(['Survived','Name','Cabin', 'Ticket'], axis=1)
    # df = df.drop(['Name', 'Cabin', 'Ticket'], axis=1)
    df.to_csv('./over_load.csv', index=False)




def ok_data():
    sns.set_style('whitegrid')
    plt.figure(figsize=(20, 10))
    # df = pd.read_csv('../data/train.csv', index_col='PassengerId')
    df = pd.read_csv('../data/test.csv', index_col='PassengerId')
    # print(df)
    print(df.columns)

    mask = df.Fare < 1000
    # 가족 유무에 대한 컬럼
    df = df[mask]  # 조건에 충족하는 값만 출력된다.
    df['fm'] = df.SibSp + df.Parch
    # 나이별로 구분
    # 결측치 0 혹은 평균 값
    # 평균 값 = df['Age'].mean()
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    # df['Age'] = df['Age'].fillna(0)
    df['a0'] = False
    df['a1'] = False
    df['a2'] = False
    df['a3'] = False
    df['a4'] = False
    df['a5'] = False
    df['a6'] = False
    df['a7'] = False
    df['a8'] = False

    def func(data):
        print(data.Age, data.a0, data.a1)
        if data.Age >= 0.0 and data.Age < 10.0:
            data.a0 = True
            return data
        elif data.Age >= 10.0 and data.Age < 20.0:
            data.a1 = True
            return data
        elif data.Age >= 20.0 and data.Age < 30.0:
            data.a2 = True
            return data
        elif data.Age >= 30.0 and data.Age < 40.0:
            data.a3 = True
            return data
        elif data.Age >= 40.0 and data.Age < 50.0:
            data.a4 = True
            return data
        elif data.Age >= 50.0 and data.Age < 60.0:
            data.a5 = True
            return data
        elif data.Age >= 60.0 and data.Age < 70.0:
            data.a6 = True
            return data
        elif data.Age >= 70.0 and data.Age < 80.0:
            data.a7 = True
            return data
        elif data.Age >= 80.0 and data.Age < 90.0:
            data.a8 = True
            return data
        return data




    df[['Age', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8']] = df[['Age', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8']].apply(func, axis=1)


    def func(data):
        if data >= 0.0 and data < 10.0:
            return 0
        elif data >= 10.0 and data < 20.0:
            return 10
        elif data >= 20.0 and data < 30.0:
            return 20
        elif data >= 30.0 and data < 40.0:
            return 30
        elif data >= 40.0 and data < 50.0:
            return 40
        elif data >= 50.0 and data < 60.0:
            return 50
        elif data >= 60.0 and data < 70.0:
            return 60
        elif data >= 70.0 and data < 80.0:
            return 70
        elif data >= 80.0 and data < 90.0:
            return 80



    # 나이대 구분
    # df['age_group'] = df['Age'].apply(func)

#    df.to_csv('./sample.csv')
    # 기항지에 따른 티켓 값 분류
    for name, group in df.groupby('Embarked'):
        #  list(df.groupby('Embarked'))
        # 키 값
        print(name)
        # 실제 데이터
        # print(group)
    # C, Q, S
    df.loc[df['Embarked'] == 'C', 'Embarked'] = 0.0
    df.loc[df['Embarked'] == 'Q', 'Embarked'] = 1.0
    df.loc[df['Embarked'] == 'S', 'Embarked'] = 2.0
    df.loc[df['Embarked'] == None, 'Embarked'] = 3.0
    df.loc[df['Embarked'].isnull(), 'Embarked'] = 3.0
    df.loc[df['Embarked'] == '', 'Embarked'] = 3.0

    # 성별
    df.loc[df['Sex'] == 'male', 'Sex'] = 0.0
    df.loc[df['Sex'] == 'female', 'Sex'] = 1.0

    # 가족 유무
    df.loc[df['fm'] > 0, 'fm'] = 1
    df.loc[df['fm'] <= 0, 'fm'] = 0


    '''
    'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
           'Fare', 'Cabin', 'Embarked'
    '''
    # 필요없는 데이터 제거
    # df = df.drop(['Survived', 'Name', 'Cabin', 'Ticket',  'SibSp', 'Parch'], axis=1)
    df = df.drop(['Name', 'Cabin', 'Ticket',  'SibSp', 'Parch'], axis=1)
    df.to_csv('./ok_data_test.csv', index=False)


def train_data():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    X_train = pd.read_csv('./ok_data.csv')
    # X_train = X_train.drop(['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8'], axis=1)
    # X_train = pd.read_csv('./over_load.csv', index_col=False)
    y_train = pd.read_csv('./y_train.csv')

    # X_train.isnull()

    # X_pre = pd.read_csv('./over_load_test.csv', index_col=False)
    X_pre = pd.read_csv('./ok_data_test.csv')
    # X_pre = X_pre.drop(['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=0)

    forest = RandomForestClassifier(criterion='entropy', n_estimators=100, random_state=0, class_weight='balanced')
    # print(X_train.isnull().sum())
    # print(y_train.sum())
    forest.fit(X_train, y_train)

    # In[29]:

    print("훈련 세트 정확도 : {:.3f}".format(forest.score(X_train, y_train)))
    print("테스트 세트 정확도 : {:.3f}".format(forest.score(X_test, y_test)))
    #predictions = forest.predict(X_pre)
    #print(predictions)

    gbrt = GradientBoostingClassifier(random_state=4, learning_rate=0.1, max_depth=3)
    gbrt.fit(X_train, y_train)

    print("훈련 세트 정확도 : {:.3f}".format(gbrt.score(X_train, y_train)))
    print("테스트 세트 정확도 : {:.3f}".format(gbrt.score(X_test, y_test)))





    def plot_feature_importances_cancer(model):
        n_features = X_train.shape[1]
        plt.barh(range(n_features), model.feature_importances_, align='center')
        plt.yticks(np.arange(n_features), X_train.columns)
        plt.xlabel("features importances")
        plt.ylabel("features")
        plt.ylim(-1, n_features)
        plt.show()

  #  X_combined = np.vstack((X_train, X_test))
  #  y_combined = np.hstack((y_train, y_test))
  #  print(X_combined)
   # print(y_combined)

    # plot_feature_importances_cancer(forest)
  #  plot_decision_regions(X_train, y_train, classifier=forest)


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    from matplotlib.colors import ListedColormap
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.aragne(x1_min, x1_max, resolution),
                           np.aragne(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array[xx1.ravel(), xx2.ravel().T])
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, X, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
                    x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl,
                    edgecolor='black'
                    )
    if test_idx:
        X_test, y_test =X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='', edgecolors='black', alpha=1.0,
                    linewidths=1, marker='o',
                    s=100, label='test set'
                    )

def test_():
    from sklearn import datasets
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    '''
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    print(X)
    print(y)
    '''
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    X_train = pd.read_csv('./ok_data.csv')
    # X_train = X_train.drop(['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8'], axis=1)
    # X_train = pd.read_csv('./over_load.csv', index_col=False)
    y_train = pd.read_csv('./y_train.csv')

    # X_train.isnull()

    # X_pre = pd.read_csv('./over_load_test.csv', index_col=False)
    X_pre = pd.read_csv('./ok_data_test.csv')
    # X_pre = X_pre.drop(['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=0)


    sc = StandardScaler()
    sc.fit(X_train)
    print(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    print(X_train_std)
    X_test_std = sc.transform(X_test)
    svm = SVC(kernel='rbf', random_state=1, gamma=1/14, C=1.0)
    svm.fit(X_train_std, y_train)
    print(svm.score(X_train_std,  y_train))
    print(svm.score(X_test_std, y_test))




    min_on_training = X_train.min(axis=0)

    # 훈련 세트에서 특성별 (최댓값 - 최솟값) 범위 계산
    range_on_training = (X_train - min_on_training).max(axis=0)

    # 훈련 데이터에 최솟값을 빼고 범위로 나누면
    # 각 특성에 대해 최솟값은 0, 최댓값은 1입니다.

    X_train_scaled = (X_train - min_on_training) / range_on_training

    print("특성별 최소 값 :\n{}".format(X_train_scaled.min(axis=0)))
    print("특성별 최대 값 :\n{}".format(X_train_scaled.max(axis=0)))

    # 테스트 세트에도 같은 작업을 작용하지만
    # 훈련 세트에서 계산한 최솟값과 범위를 사용합니다.

    X_test_scaled = (X_test - min_on_training) / range_on_training

    svc = SVC()
    svc.fit(X_train_scaled, y_train)

    print("훈련 세트 정확도:{:.3f}".format(svc.score(X_train_scaled, y_train)))
    print("테스트 세트 정확도:{:.3f}".format(svc.score(X_test_scaled, y_test)))

    svc = SVC(C=1000)
    svc.fit(X_train_scaled, y_train)
    print("훈련 세트 정확도:{:.3f}".format(svc.score(X_train_scaled, y_train)))
    print("테스트 세트 정확도:{:.3f}".format(svc.score(X_test_scaled, y_test)))

def jy_():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    X_train = pd.read_csv('../data/train_over22.csv')
    X_train = X_train.drop(['Survived','Embarked_C','Embarked_S','Embarked_Q'], axis=1)
    # X_train = pd.read_csv('./over_load.csv', index_col=False)
    y_train = pd.read_csv('./y_train.csv')

    # Cabin의 키 값
    for name, group in X_train.groupby('Cabin'):
        #  list(df.groupby('Embarked'))
        # 키 값
        print(name)
        # 실제 데이터
        # print(group)
    # C, Q, S
    print(X_train['Cabin'])
    X_train.loc[X_train['Cabin'] == 'A', 'Cabin'] = 0.0
    X_train.loc[X_train['Cabin'] == 'B', 'Cabin'] = 1.0
    X_train.loc[X_train['Cabin'] == 'C', 'Cabin'] = 2.0
    X_train.loc[X_train['Cabin'] == 'D', 'Cabin'] = 3.0
    X_train.loc[X_train['Cabin'] == 'E', 'Cabin'] = 4.0
    X_train.loc[X_train['Cabin'] == 'F', 'Cabin'] = 5.0
    X_train.loc[X_train['Cabin'] == 'G', 'Cabin'] = 6.0
    X_train.loc[X_train['Cabin'] == 'T', 'Cabin'] = 7.0
    X_train.loc[X_train['Cabin'].isnull(), 'Cabin'] = 8.0

    print(X_train.isnull().sum())
    print(y_train.sum())

    # X_pre = pd.read_csv('./over_load_test.csv', index_col=False)
    # X_pre = pd.read_csv('../data/test_fit_ff.csv')
    # X_pre = X_pre.drop(['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=0)

    forest = RandomForestClassifier( n_estimators=11, random_state=0, class_weight='balanced')
    # print(X_train.isnull().sum())
    # print(y_train.sum())
    # print(X_train)
    # print(y_train)
    forest.fit(X_train, y_train)

    # In[29]:

    print("훈련 세트 정확도 : {:.3f}".format(forest.score(X_train, y_train)))
    print("테스트 세트 정확도 : {:.3f}".format(forest.score(X_test, y_test)))
    # predictions = forest.predict(X_pre)
    # print(predictions)

    gbrt = GradientBoostingClassifier(random_state=4, learning_rate=0.1, max_depth=3)
    gbrt.fit(X_train, y_train)

    print("훈련 세트 정확도 : {:.3f}".format(gbrt.score(X_train, y_train)))
    print("테스트 세트 정확도 : {:.3f}".format(gbrt.score(X_test, y_test)))


    # svm
    sc = StandardScaler()
    sc.fit(X_train)
    print(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    print(X_train_std)
    X_test_std = sc.transform(X_test)
    svm = SVC(kernel='rbf', random_state=1, gamma=1 / 14, C=1.0)
    svm.fit(X_train_std, y_train)
    print(svm.score(X_train_std, y_train))
    print(svm.score(X_test_std, y_test))

    min_on_training = X_train.min(axis=0)

    # 훈련 세트에서 특성별 (최댓값 - 최솟값) 범위 계산
    range_on_training = (X_train - min_on_training).max(axis=0)

    # 훈련 데이터에 최솟값을 빼고 범위로 나누면
    # 각 특성에 대해 최솟값은 0, 최댓값은 1입니다.

    X_train_scaled = (X_train - min_on_training) / range_on_training

    print("특성별 최소 값 :\n{}".format(X_train_scaled.min(axis=0)))
    print("특성별 최대 값 :\n{}".format(X_train_scaled.max(axis=0)))

    # 테스트 세트에도 같은 작업을 작용하지만
    # 훈련 세트에서 계산한 최솟값과 범위를 사용합니다.

    X_test_scaled = (X_test - min_on_training) / range_on_training

    svc = SVC()
    svc.fit(X_train_scaled, y_train)

    print("훈련 세트 정확도:{:.3f}".format(svc.score(X_train_scaled, y_train)))
    print("테스트 세트 정확도:{:.3f}".format(svc.score(X_test_scaled, y_test)))

    svc = SVC(C=1000)
    svc.fit(X_train_scaled, y_train)
    print("훈련 세트 정확도:{:.3f}".format(svc.score(X_train_scaled, y_train)))
    print("테스트 세트 정확도:{:.3f}".format(svc.score(X_test_scaled, y_test)))

if __name__ == '__main__':
    # titanic()
    # over_data()
    # ok_data()
    # train_data()
    test_()
    # jy_()