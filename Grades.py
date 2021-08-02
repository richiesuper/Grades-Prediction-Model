import pandas as pd
import numpy as np

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