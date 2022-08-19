import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import DropFeatures, OneHotEncodingNames, OrdinalFeature, MinMaxWithFeatNames
from sklearn.pipeline import Pipeline
import joblib
from joblib import load 

dados = pd.read_csv('https://raw.githubusercontent.com/alura-tech/alura-tech-pos-data-science-credit-scoring-streamlit/main/df_clean.csv')

st.markdown('<style>div[role="listbox"] ul{background-color: #6e42ad}; </style>', unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; '> Formulário para Solicitação de Cartão de Crédito 🤑</h1>", unsafe_allow_html = True)

st.warning('Preencha o formulário com todos os seus dados pessoais e clique no botão **ENVIAR** no final da página.')

st.write('### Idade')
input_idade = float(st.slider('Selecione sua idade', 18, 100))

st.write('### Nível de escolaridade')
input_grau_escolaridade = st.selectbox('Qual é o seu grau de escolaridade?', dados['Grau_escolaridade'].unique())

st.write('### Estado civil')
input_estado_civil = st.selectbox('Qual é o seu estado civil?', dados['Estado_civil'].unique())

st.write('### Família')
membros_familia = float(st.slider('Selecione quantos membros tem na sua família', 1, 20))

st.write('### Carro próprio')
input_carro_proprio = st.radio('Você possui um automóvel?', ['Sim', 'Não'])
input_carro_proprio_dict = {'Sim':1, 'Não':0}
input_carro_proprio = input_carro_proprio_dict.get(input_carro_proprio)

st.write('### Casa própria')
input_casa_propria = st.radio('Você possui uma propriedade?', ['Sim', 'Não'])
input_casa_propria_dict = {'Sim':1, 'Não':0}
input_casa_propria = input_casa_propria_dict.get(input_casa_propria)

st.write('### Tipo de residência')
input_tipo_moradia = st.selectbox('Qual é o seu tipo de moradia?', dados['Moradia'].unique())

st.write('### Categoria de renda')
input_categoria_renda = st.selectbox('Qual é a sua categoria de renda?', dados['Categoria_de_renda'].unique())

st.write('### Ocupação')
input_ocupacao = st.selectbox('Qual é a sua ocupação?', dados['Ocupacao'].unique())

st.write('### Experiência')
input_tempo_experiencia = float(st.slider('Qual é o seu tempo de experiência?', 0, 30))

st.write('### Rendimentos')
input_rendimentos = float(st.number_input('Digite o seu rendimento anual (em reais) e pressione ENTER para confirmar', 0))

st.write('### Telefone corporativo')
input_telefone_trabalho = st.radio('Você possui um telefone corporativo?', ['Sim', 'Não'])
input_telefone_trabalho_dict = {'Sim':1, 'Não':0}
input_telefone_trabalho = input_telefone_trabalho_dict.get(input_telefone_trabalho)

st.write('### Telefone fixo')
input_telefone = st.radio('Você possui um telefone fixo?', ['Sim', 'Não'])
input_telefone_dict = {'Sim':1, 'Não':0}
input_telefone = input_telefone_dict.get(input_telefone)

st.write('### Email')
input_email = st.radio('Você tem um email?', ['Sim', 'Não'])
input_email_dict = {'Sim':1, 'Não':0}
input_email = input_email_dict.get(input_email) 

novo_cliente = [0, 
                input_carro_proprio,
                input_casa_propria,
                input_telefone_trabalho,
                input_telefone,
                input_email,
                membros_familia,
                input_rendimentos,
                input_idade,
                input_tempo_experiencia,
                input_categoria_renda,
                input_grau_escolaridade,
                input_estado_civil,
                input_tipo_moradia,
                input_ocupacao,
                0
]

def data_split(df, test_size):
    SEED=1561651
    treino_df, teste_df = train_test_split(df, test_size=test_size, random_state=SEED)
    return treino_df.reset_index(drop=True), teste_df.reset_index(drop=True)

treino_df, teste_df = data_split(dados, 0.2)


cliente_predict_df = pd.DataFrame([novo_cliente], columns=teste_df.columns)

teste_novo_cliente = pd.concat([teste_df, cliente_predict_df], ignore_index=True)

#Pipeline
def pipeline_teste(df):

    pipeline = Pipeline([
        ('feature_dropper', DropFeatures()),
        ('OneHotEncoding', OneHotEncodingNames()),
        ('ordinal_feature', OrdinalFeature()),
        ('min_max_scaler', MinMaxWithFeatNames()),
    ])
    df_pipeline = pipeline.fit_transform(df)
    return df_pipeline

teste_novo_cliente = pipeline_teste(teste_novo_cliente)

cliente_pred = teste_novo_cliente.drop(['Mau'], axis=1)

if st.button('Enviar'):
    model = joblib.load('modelo/xgb.joblib')
    final_pred = model.predict(cliente_pred)
    if final_pred[-1] == 0:
        st.success('### Parabéns! Você teve o cartão de crédito aprovado')
        st.balloons() 
    else: 
        st.error('### Infelizmente, não podemos liberar crédito para você agora!')

