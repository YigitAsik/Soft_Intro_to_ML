import os
os.getcwd()  # hangi working directorydeyim.
from data.YigasHelpers import *  # kendi fonksiyonlarımı ve lazım olan kütüphaneleri importladım.


# dataframe wide ve long olsa bile tüm gözlem birimlerini ve sütunları görmek için ayarlamalar.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# datayı pandas yardımı ile okudum.
titanic = pd.read_csv('data/burak_hoca_ornek/titanic.csv')

# sibsp - Number of Siblings/Spouses Aboard
# parch - Number of Parents/Children Aboard
# embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

# df diye bir objeye kopyaladım.
df = titanic.copy()

# ilk 5 gözlemine bakıyorum.
df.head(3)

# sütun isimlerini büyük harfler yaptım, ='in sağ tarafı bir for loop aslında, liste içerisinde yazılmış hali.
# list comprehension deniyor buna.
df.columns = [col.upper() for col in df.columns]

# Hedef değişkenimin seviyelerinin dağılımını öğrendim.
df['SURVIVED'].value_counts() * 100 / len(df)

# Veriyi train ve test olacak şekilde ikiye ayırıyorum, bunu yaparken hedef değişkenin dağılımı bu train ve test gruplarında
# eşit olsun diye 'stratify' verdim.
train, test = train_test_split(df, test_size=.2, stratify=df.SURVIVED, random_state=0)

# train ile çalışacağız.
train.info() # sütunlara, tuttukları veri tiplerine bakıyorum.

train.describe([.05, .25, .5, .75, .95, .99]).T # sütundaki değerlerin dağılımı hakkında bir ön fikir edinmek için

train.isnull().sum() # sütun sütun boş değer toplamlarına bakıyorum.

missing_values_table(train) # bu değerlerin dataya oranına bakıyorum.

train.drop("CABIN", axis=1, inplace=True) # CABIN sütununun %77.5'i boş olduğundan sütunu dropluyorum
test.drop("CABIN", axis=1, inplace=True) # CABIN sütununun %77.5'i boş olduğundan sütunu dropluyorum

# Yaş değişkeninin dağılımı
fig = plt.figure(figsize=(10,8))
g = sns.distplot(x=train["AGE"], kde=False, color="Red", hist_kws=dict(edgecolor="black", linewidth=2))
g.set_title("Column: AGE")
g.xaxis.set_minor_locator(AutoMinorLocator(5))
g.yaxis.set_minor_locator(AutoMinorLocator(5))
g.tick_params(which="both", width=2)
g.tick_params(which="major", length=7)
g.tick_params(which="minor", length=4)
plt.show()

fig = plt.figure(figsize=(10,8))
g = sns.kdeplot(x=train["AGE"], color="Red", shade=True)
g.set_title("Column: Age")
g.xaxis.set_minor_locator(AutoMinorLocator(5))
g.yaxis.set_minor_locator(AutoMinorLocator(5))
g.tick_params(which="both", width=2)
g.tick_params(which="major", length=7)
g.tick_params(which="minor", length=4)
plt.show()

train.groupby('SURVIVED').agg({'AGE': 'mean'})

fig = plt.figure(figsize=(10,8))
g = sns.kdeplot(x=train["AGE"], hue=train["SURVIVED"], shade=True)
g.set_title("Column: Age")
g.xaxis.set_minor_locator(AutoMinorLocator(5))
g.yaxis.set_minor_locator(AutoMinorLocator(5))
g.tick_params(which="both", width=2)
g.tick_params(which="major", length=7)
g.tick_params(which="minor", length=4)
plt.show()

train["AGE"].fillna(train["AGE"].mean(), inplace=True) # yaş sütunundaki boş değişkenleri ortalama ile doldurdum.
test["AGE"].fillna(train["AGE"].mean(), inplace=True) # yaş sütunundaki boş değişkenleri ortalama ile doldurdum.
# Farklı doldurma yöntemleri var: random sample imputation, multiple imputation, regression imputation
# fakat bunlara girmeyeceğiz şimdilik.
train.isnull().sum().sort_values(ascending=False)
test.isnull().any()

train.dropna(inplace=True)
train.isnull().any()

train.shape
train.info()

cat_cols = [col for col in train.columns if str(train[col].dtypes) == "object"]
num_but_cat = [col for col in train.columns if train[col].nunique() < 5 and train[col].dtypes in ["int64", "int32", "float64", "float32"]]
cat_but_car = [col for col in train.columns if train[col].nunique() > 20 and str(train[col].dtypes) in ["category", "object"]]

cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]

num_cols = [col for col in train.columns if train[col].dtypes in ["int", "float"]]
num_cols = [col for col in num_cols if col not in cat_cols]

cat_cols.append('SIBSP')
cat_cols.append('PARCH')

for col in cat_cols:
    if col != 'SURVIVED':
        print(train.groupby(col).agg('SURVIVED').mean())

train.PCLASS.dtypes

train.groupby('EMBARKED').agg({'PCLASS': 'mean'})

train.groupby(["EMBARKED", "PCLASS"]).agg({"SURVIVED": "mean"})

train.PARCH.value_counts()

cat_summary(train, 'PARCH')

for col in cat_cols:
    if col != 'SURVIVED':
        print('####')
        print(str(col))
        target_summary_with_cat(train, 'SURVIVED', col)

for col in num_cols:
    target_summary_with_num(train, 'SURVIVED', col)

rare_analyser(train, 'SURVIVED', cat_cols)

train['PARCH'].dtypes
train['PARCH'] = train['PARCH'].astype(object)
train.SIBSP.dtypes
train['SIBSP'] = train['SIBSP'].astype(object)
train = rare_encoder(train, .05)

test['PARCH'] = test['PARCH'].astype(object)
test['SIBSP'] = test['SIBSP'].astype(object)
test = rare_encoder(test, .05)

rare_analyser(test, 'SURVIVED', cat_cols)
rare_analyser(train, 'SURVIVED', cat_cols)

cat_colmns = [col for col in cat_cols if col != 'SURVIVED']

train = one_hot_encoder(train, cat_colmns, drop_first=False)
test = one_hot_encoder(test, cat_colmns, drop_first=False)

train.shape[1] == test.shape[1]

train.head(2)

mms = MinMaxScaler()

mms.fit(train[['FARE']])
scaled = mms.transform(train[['FARE']])
train['FARE'] = scaled

scaled = mms.transform(test[['FARE']])
test['FARE'] = scaled

mms.fit(train[['AGE']])
scaled = mms.transform(train[['AGE']])
train['AGE'] = scaled

scaled = mms.transform(test[['AGE']])
test['AGE'] = scaled

train.head(2)

X_train = train.drop(['PASSENGERID', 'NAME', 'TICKET', 'SURVIVED'], axis=1)
X_test = test.drop(['PASSENGERID', 'NAME', 'TICKET', 'SURVIVED'], axis=1)

y_train = train['SURVIVED']
y_test = test['SURVIVED']

from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.ensemble import RandomForestClassifier

cft = DecisionTreeClassifier(random_state=0)

cv_results = cross_validate(cft,
                            X_train, y_train,
                            cv=5,
                            scoring=['f1', 'precision', 'recall', 'accuracy'])

cv_results['test_accuracy'].mean()
cv_results['test_precision'].mean()
cv_results['test_recall'].mean()
cv_results['test_f1'].mean()

cft.get_params()

cft_params = {'max_depth': range(3, 11),
               'min_samples_split': range(6, 25)}


cft_grid_search = GridSearchCV(cft,
                             cft_params,
                             cv=5,
                             n_jobs=-1,
                             verbose=1)

cft_grid_search.fit(X_train, y_train)

cft_grid_search.best_params_

cft_fin = DecisionTreeClassifier(**cft_grid_search.best_params_, random_state=0).fit(X_train, y_train)

val_curve_params(cft_fin, X_train, y_train, 'max_depth', range(1,11), scoring='accuracy')

y_pred = cft_fin.predict(X_test)
print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 3)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 3)}")
print(f"F1: {round(f1_score(y_pred,y_test), 3)}")

fig = plt.figure(figsize=(8, 8))
cm = confusion_matrix(y_test, y_pred, labels=cft_fin.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cft_fin.classes_)
disp.plot()
plt.show()

import pydotplus

def tree_graph(model, col_names, file_name):
    tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)

tree_graph(model=cft_fin, col_names=X_train.columns, file_name='cft_fin_titanic.png')

import eli5
from eli5.sklearn import PermutationImportance

from sklearn.inspection import permutation_importance

r = permutation_importance(cft_fin, X_train, y_train,
                           n_repeats=30,
                           random_state=0)

for i in r.importances_mean.argsort()[::-1]:
     if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
         print(f" {X_train.columns[i]:<8}"
               f" {r.importances_mean[i]:.3f}"
               f" +/- {r.importances_std[i]:.3f}")

r = permutation_importance(cft_fin, X_test, y_test,
                           n_repeats=30,
                           random_state=0)

for i in r.importances_mean.argsort()[::-1]:
     if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
         print(f" {X_train.columns[i]:<8}"
               f" {r.importances_mean[i]:.3f}"
               f" +/- {r.importances_std[i]:.3f}")

r_multi = permutation_importance(cft_fin, X_test, y_test,
                                 n_repeats=30,
                                 random_state=0,
                                 scoring=['accuracy', 'precision', 'recall', 'f1'])

for metric in r_multi:
     print(f"{metric}")
     r = r_multi[metric]
     for i in r.importances_mean.argsort()[::-1]:
         if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
             print(f"    {X_train.columns[i]:<8}"
                   f" {r.importances_mean[i]:.3f}"
                   f" +/- {r.importances_std[i]:.3f}")

from pdpbox import pdp, get_dataset, info_plots

feature_names = X_test.columns.tolist()

pdp_age = pdp.pdp_isolate(model=cft_fin, dataset=X_test, model_features=feature_names, feature="AGE")

pdp.pdp_plot(pdp_age, "Age")
plt.show()

features_to_plot = ["AGE", "FARE"]
inter1 = pdp.pdp_interact(model=cft_fin, dataset=X_test, model_features=feature_names, features=features_to_plot)

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')
plt.show()

import shap

explainer = shap.TreeExplainer(cft_fin)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], X_test)

explainer = shap.TreeExplainer(cft_fin)
shap_values = explainer.shap_values(X_test)
shap.dependence_plot("AGE", shap_values[1], X_test)

# Each dot represents a row of the data
# The horizontal location is the actual value from the dataset,
# The vertical location shows that having that value did to prediction

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=0)

rf_params = {"max_depth": [5, 7, 10, 12, None],
             "max_features": [5, 7, 10, "sqrt", "auto"],
             "min_samples_split": [6, 8, 15, 20, 25],
             "n_estimators": [300, 500, 800]}

rf_grid_search = GridSearchCV(
    rf,
    rf_params,
    cv=5,
    n_jobs=-1,
    verbose=1
)

rf_grid_search.fit(X_train, y_train)

rf_grid_search.best_params_

rf_fin = RandomForestClassifier(**rf_grid_search.best_params_, random_state=0).fit(X_train, y_train)

y_pred = rf_fin.predict(X_test)
print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 3)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 3)}")
print(f"F1: {round(f1_score(y_pred,y_test), 3)}")

fig = plt.figure(figsize=(8, 8))
cm = confusion_matrix(y_test, y_pred, labels=rf_fin.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_fin.classes_)
disp.plot()
plt.show()

