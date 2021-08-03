import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

original_df = pd.read_csv('test_scores.csv')

# school binary dummies
school_dummies = pd.get_dummies(original_df.school)
df_school = pd.concat([school_dummies, original_df], axis=1, join="inner")
df_school.drop('school', inplace=True, axis=1)

# school_setting ordinal
df_school['school_setting'] = df_school['school_setting'].astype('category')
df_school['school_setting'] = df_school['school_setting'].cat.reorder_categories(['Rural', 'Suburban', 'Urban'])
df_school['school_setting'] = df_school['school_setting'].cat.codes
df_school_setting = df_school

# school_type binary
df_school_setting['school_type'].replace('Non-public', 0, inplace=True)
df_school_setting['school_type'].replace('Public', 1, inplace=True)
df_school_setting_type = df_school_setting

# teaching_method binary
df_school_setting_type['teaching_method'].replace('Standard', 0, inplace=True)
df_school_setting_type['teaching_method'].replace('Experimental', 1, inplace=True)
df_school_setting_type_method = df_school_setting_type

# student_id omit
df_school.drop('student_id', inplace=True, axis=1)

# classroom binary dummies
classroom_dummies = pd.get_dummies(df_school_setting_type_method.classroom)
df_school_setting_type_method_classroom = pd.concat([classroom_dummies, df_school_setting_type_method], axis=1, join="inner")
df_school_setting_type_method_classroom.drop('classroom', inplace=True, axis=1)

# gender binary dummies
gender_dummies = pd.get_dummies(df_school_setting_type_method.gender)
df_school_setting_type_method_classroom_gender = pd.concat([gender_dummies, df_school_setting_type_method_classroom], axis=1, join="inner")
df_school_setting_type_method_classroom_gender.drop('gender', inplace=True, axis=1)

# lunch binary
df_school_setting_type_method_classroom_gender['lunch'].replace('Does not qualify', 0, inplace=True)
df_school_setting_type_method_classroom_gender['lunch'].replace('Qualifies for reduced/free lunch', 1, inplace=True)
df_school_setting_type_method_classroom_gender_lunch = df_school_setting_type_method_classroom_gender

# df
df = df_school_setting_type_method_classroom_gender_lunch

# Getting all the X Headers
X_headers = []
for i in df.columns[:-1]:
    X_headers += [i]
gender = X_headers[:2]
classroom = X_headers[2:99]
school = X_headers[99:122]


X_headers.remove('teaching_method')

X = df.loc[:, X_headers]
y = df.loc[:, ['posttest']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = pd.DataFrame(regressor.predict(X_test), columns=['posttest'])

RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print(RMSE)
# w/ all = 2.88474
# w/o Gender =  2.89301
# w/o Classroom = 3.06543
# w/o School = 2.8847
# w/o school_setting = 2.88471
# w/o n_student = 2.88475
# w/o lunch = 2.9272
# w/o teaching method = 2.88476
