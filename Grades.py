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

# school_setting binary dummies
school_setting_dummies = pd.get_dummies(df_school.school_setting)
df_school_setting = pd.concat([school_setting_dummies, df_school], axis=1, join="inner")
df_school_setting.drop('school_setting', inplace=True, axis=1)

# school_type binary dummies
school_type_dummies = pd.get_dummies(df_school_setting.school_type)
df_school_setting_type = pd.concat([school_type_dummies, df_school_setting], axis=1, join="inner")
df_school_setting_type.drop('school_type', inplace=True, axis=1)

# teaching_method binary dummies
teaching_method_dummies = pd.get_dummies(df_school_setting_type.teaching_method)
df_school_setting_type_method = pd.concat([teaching_method_dummies, df_school_setting_type], axis=1, join="inner")
df_school_setting_type_method.drop('teaching_method', inplace=True, axis=1)

# classroom binary dummies
classroom_dummies = pd.get_dummies(df_school_setting_type_method.classroom)
df_school_setting_type_method_classroom = pd.concat([classroom_dummies, df_school_setting_type_method], axis=1, join="inner")
df_school_setting_type_method_classroom.drop('classroom', inplace=True, axis=1)

# gender binary dummies
gender_dummies = pd.get_dummies(df_school_setting_type_method.gender)
df_school_setting_type_method_classroom_gender = pd.concat([gender_dummies, df_school_setting_type_method_classroom], axis=1, join="inner")
df_school_setting_type_method_classroom_gender.drop('gender', inplace=True, axis=1)

# lunch binary dummies
lunch_dummies = pd.get_dummies(df_school_setting_type_method_classroom_gender.lunch)
df_school_setting_type_method_classroom_gender_lunch = pd.concat([lunch_dummies, df_school_setting_type_method_classroom_gender], axis=1, join="inner")
df_school_setting_type_method_classroom_gender_lunch.drop('lunch', inplace=True, axis=1)

# df
df = df_school_setting_type_method_classroom_gender_lunch
df.drop('student_id', inplace=True, axis=1)

# Getting all the X Headers
X_headers = []
for i in df.columns[:-1]:
    X_headers += [i]
# gender = X_headers[:2]
# classroom = X_headers[2:99]
# school = X_headers[99:122]

X = df.loc[:, X_headers]
y = df.loc[:, ['posttest']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8950)
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