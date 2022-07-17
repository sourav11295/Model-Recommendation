import gradio as gr
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def read(file,dep,ord):
    df = pd.read_csv(file.name)
    cat = list()
    dep_type = str(df.dtypes[dep])
    for col in df.columns.values:
        if str(df.dtypes[col]) == 'bool' or str(df.dtypes[col]) == 'object':
            cat.append(col)
    new_df = df.dropna(axis=0)
    if ord == "" and (dep_type == 'bool' or dep_type == 'object'):
        ord = list()
        ord.append(dep)
    elif ord == "":
        ord = list()
    else:
        pass
    if len(ord)!=0:
        le = LabelEncoder()
        new_df[ord] = new_df[ord].apply(lambda col: le.fit_transform(col))
    nom = list(set(cat).difference(set(ord)))
    if len(nom) == 0:
        pass
    else:
        ohe_df = pd.get_dummies(new_df[nom], drop_first=True)
        new_df.drop(columns=nom, axis=1,inplace=True)
        new_df = pd.concat([new_df,ohe_df],axis=1)
    if dep_type == 'bool' or dep_type == 'object':
        text = "classification"
        result = classification(new_df,dep)
    else:
        text = "regression"
        result = regression(new_df,dep)
    return df.head(5),new_df.head(5),result, text, cat, ord, nom
    
def classification(df,dep):
    X = df.drop(dep,axis=1)
    y = df[dep]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    scale = StandardScaler()

    pipe = Pipeline(steps=[('scale',scale),('classification','pass')])

    parameters = [
        {
            'classification':[LogisticRegression()],
        },
        {
            'classification':[RandomForestClassifier()],
        },
        {
            'classification':[DecisionTreeClassifier()],
        },
        {
            'classification':[SVC()],
        },
        {
            'classification':[KNeighborsClassifier(n_neighbors=5)],
        },
    ]

    search = GridSearchCV(pipe, param_grid=parameters, n_jobs=-1, scoring='accuracy')
    search.fit(X_train,y_train)

    result = pd.DataFrame(search.cv_results_)[['params','rank_test_score','mean_test_score']]

    result['mean_test_score']= (result['mean_test_score'])*100
    result = result.astype({'params': str})

    result.sort_values('rank_test_score',inplace=True)
    return result

def regression(df,dep):
    X = df.drop(dep,axis=1)
    y =df[dep]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    scale = StandardScaler()

    pipe = Pipeline(steps=[('scale',scale),('regression','pass')])

    parameters = [
        {
            'regression':[LinearRegression()]
        },
        {
            'regression':[RandomForestRegressor()],
        },
        {
            'regression':[DecisionTreeRegressor()],
        },
        {
            'regression':[SVR()],
        },
    ]

    search = GridSearchCV(pipe, param_grid=parameters, cv=5, n_jobs=-1, scoring='neg_mean_absolute_percentage_error')
    search.fit(X_train,y_train)

    result = pd.DataFrame(search.cv_results_)[['params','rank_test_score','mean_test_score']]
    
    result['mean_test_score']= (result['mean_test_score']+1)*100
    result = result.astype({'params': str})
    
    result.sort_values('rank_test_score',inplace=True)
    return result
    

with gr.Blocks() as demo:
    gr.Markdown("Model Recommendation App **Upload** file to see the output.")
    with gr.Column():
        with gr.Row():
            file = gr.File(label="Upload File(Comma Separated)")
            dep = gr.Textbox(label="Dependent Variable(Variable as in the file)")
            ord = gr.Textbox(label="Ordinal Variables(Seperate with a comma)")
            submit = gr.Button("Submit")
        text = gr.Text(label="Suitable Algorithm")
        other1 = gr.Text(label="Categorical Variables")
        other2 = gr.Text(label="LabelEncoded Vairables")
        other3 = gr.Text(label="OneHotEncoded Variables")
        with gr.Row():
            org = gr.DataFrame(overflow_row_behaviour="paginate", label="Original Data")
            converted = gr.DataFrame(overflow_row_behaviour="paginate", label="Transformed Data")
        result = gr.DataFrame(label="Result")
    submit.click(fn=read, inputs=[file,dep,ord], outputs=[org,converted,result,text,other1,other2,other3])
demo.launch()
