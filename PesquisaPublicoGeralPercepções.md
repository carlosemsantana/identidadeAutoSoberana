### **Análise de dados da pesquisa sobre vazamentos de dados**
#### Inteligência Artificial e Identidade Auto Soberana: Defesa Contra Vazamentos de Dados
Autor: Carlos Eugênio Moreira de Santana <br>
Instituição: POLI USP Pró <br>
Orientador: Felipe Francisco Nusda <br>
Última Atualização: 09 de Julho de 2025 <br>

#### **Introdução ao Notebook de Análise de Dados** ####

Este Jupyter Notebook é um guia detalhado do processo de análise dos dados coletados por meio do questionário online da pesquisa "Inteligência Artificial e Identidade Auto Soberana: Defesa Contra Vazamentos de Dados". Esta pesquisa é parte integrante do Trabalho de Conclusão de Curso (TCC) de Carlos Eugênio Moreira de Santana, desenvolvido na Escola Politécnica da USP.

O principal propósito desta pesquisa é investigar as percepções e aplicações da Inteligência Artificial (IA) e da Identidade Auto Soberana (Self-Sovereign Identity - SSI) como mecanismos eficazes de defesa e prevenção contra vazamentos de dados.

Especificamente, a análise dos dados do questionário focará em:<BR>

1) Compreender a percepção do público em geral sobre os riscos de vazamentos de dados;<BR>
2) Avaliar o nível de preocupação dos participantes com esses riscos;<BR>
3) Entender a eficácia percebida de soluções de segurança digital, com destaque para a Identidade Auto Soberana.<BR>

É importante ressaltar que a participação no questionário foi voluntária e as informações foram coletadas de forma anônima, garantindo total confidencialidade e privacidade aos participantes. Os resultados e insights gerados neste notebook servirão como a base para a seção de Resultados Preliminares do TCC, oferecendo diretrizes valiosas para a implementação e o uso eficaz dessas tecnologias no combate a vazamentos de dados.


#### **Roteiro para os Próximos Passos da Análise (com Potencial de Automação)** ####

A análise será estruturada nos seguintes passos, garantindo uma abordagem metodológica e transparente, com foco na sua potencial replicabilidade em um pipeline de processamento automatizado de dados:

**Carregamento e Exploração Inicial dos Dados**


```python
# --- Configuração Inicial ---
# Importar as bibliotecas necessárias
# Crie um ambiente isolado para o processamento dos dados
# conda create -n tcc_analise_v2 python=3.9 -y
# conda activate tcc_analise_v2
# pip install pandas numpy==1.24.4 scipy==1.10.1 nltk wordcloud textblob gensim==4.3.3 pyLDAvis==3.4.1
# Para exportar o documento final use:
# jupyter nbconvert --to markdown seu_notebook.ipynb
import pandas as pd
import numpy as np 
import re
import matplotlib.pyplot as plt
import seaborn as sns
# Dependências
# Análise de dados PLN (Processamento de textos)
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
# Para Modelagem de Tópicos (LDA)
from gensim import corpora, models
import pyLDAvis.gensim_models as gensim_models
import pyLDAvis
# Baixa as stopwords (se ainda não baixadas)
try:
    stopwords.words('portuguese')
except LookupError:
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger') # Necessário para TextBlob às vezes
```

Importação da base de dados coletada (em formato CSV ou Excel) para o ambiente do Jupyter Notebook.


```python
# Importação dos dados no formato excel, com base nessa premissa aqui definida. Formate os atributos para facilitar a manipulação.
fonte_dados = "pesquisaPublicoGeralRaw.xlsx"
# Atribui nomes mais fáceis de usar às colunas da tabela
df = pd.read_excel(fonte_dados)
```

## **Exploração Inicial dos Dados**

Verificação das dimensões do dataset, inspeção das primeiras linhas, verificação dos tipos de dados e identificação de valores ausentes.



```python
# Crie um DataFrame vazio com as colunas especificadas, para descobir quais são os atributos da fonte de dados
df_pesquisa_original = pd.DataFrame(columns=df.columns.tolist())
```


```python
df_pesquisa_original
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Carimbo de data/hora</th>
      <th>CONSENTIMENTO DO PARTICIPANTE:\nAo marcar a caixa abaixo, declaro que li e compreendi todas as informações contidas neste Termo de Consentimento Livre e Esclarecido. Fui devidamente informado(a) sobre os objetivos da pesquisa, a natureza da minha participação, as garantias de confidencialidade e privacidade, e meus direitos como participante.\nAssim, de forma livre e espontânea, concordo em participar desta pesquisa e autorizo o uso das informações fornecidas por mim através deste questionário, conforme os termos aqui descritos, para fins exclusivamente acadêmicos.</th>
      <th>Qual a sua faixa etária?</th>
      <th>Qual o seu nível de escolaridade ?</th>
      <th>Quão preocupado(a) você está com a segurança de seus dados pessoais online (ex: nome, CPF, senhas, fotos)?</th>
      <th>Com que frequência você ouve falar sobre notícias de vazamentos de dados ou ataques cibernéticos?</th>
      <th>Você já foi afetado(a) diretamente ou conhece alguém que foi afetado(a) por um vazamento de dados (ex: dados de cartão de crédito, informações pessoais, e-mail)?</th>
      <th>Em uma escala de 1 a 5, o quanto você confia nas soluções de segurança digital que utiliza atualmente (ex: antivírus, senhas fortes, biometria, autenticação em duas etapas) para proteger seus dados?</th>
      <th>Você já ouviu falar sobre o uso de Inteligência Artificial (IA) para proteger dados contra vazamentos?</th>
      <th>Em sua opinião, o quanto a Inteligência Artificial pode ser eficaz na prevenção de vazamentos de dados?</th>
      <th>Imagine uma tecnologia onde você tem total controle sobre seus próprios dados de identidade (ex: RG, CPF, diplomas, histórico médico), decidindo com quem e quando compartilhar cada informação, sem depender de uma única empresa. Você teria interesse em usar uma solução assim?</th>
      <th>Em sua opinião, o quanto é importante que as pessoas tenham mais controle sobre seus próprios dados pessoais na internet?</th>
      <th>Gostaria de adicionar algum comentário ou observação sobre segurança de dados, privacidade ou tecnologias de proteção?</th>
      <th>Pontuação</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# Analisar conteúdo das colunão que não queremos analisar
df["CONSENTIMENTO DO PARTICIPANTE:\nAo marcar a caixa abaixo, declaro que li e compreendi todas as informações contidas neste Termo de Consentimento Livre e Esclarecido. Fui devidamente informado(a) sobre os objetivos da pesquisa, a natureza da minha participação, as garantias de confidencialidade e privacidade, e meus direitos como participante.\nAssim, de forma livre e espontânea, concordo em participar desta pesquisa e autorizo o uso das informações fornecidas por mim através deste questionário, conforme os termos aqui descritos, para fins exclusivamente acadêmicos."].head()
```




    0    SIM, eu li e concordo com os termos do Termo d...
    1    SIM, eu li e concordo com os termos do Termo d...
    2    SIM, eu li e concordo com os termos do Termo d...
    3    SIM, eu li e concordo com os termos do Termo d...
    4    SIM, eu li e concordo com os termos do Termo d...
    Name: CONSENTIMENTO DO PARTICIPANTE:\nAo marcar a caixa abaixo, declaro que li e compreendi todas as informações contidas neste Termo de Consentimento Livre e Esclarecido. Fui devidamente informado(a) sobre os objetivos da pesquisa, a natureza da minha participação, as garantias de confidencialidade e privacidade, e meus direitos como participante.\nAssim, de forma livre e espontânea, concordo em participar desta pesquisa e autorizo o uso das informações fornecidas por mim através deste questionário, conforme os termos aqui descritos, para fins exclusivamente acadêmicos., dtype: object




```python
# Remover as colunas que não queremos analisar.
coluna_1 = '''CONSENTIMENTO DO PARTICIPANTE:\nAo marcar a caixa abaixo, declaro que li e compreendi todas as informações contidas neste Termo de Consentimento Livre e Esclarecido. Fui devidamente informado(a) sobre os objetivos da pesquisa, a natureza da minha participação, as garantias de confidencialidade e privacidade, e meus direitos como participante.\nAssim, de forma livre e espontânea, concordo em participar desta pesquisa e autorizo o uso das informações fornecidas por mim através deste questionário, conforme os termos aqui descritos, para fins exclusivamente acadêmicos.'''
coluna_2 = 'Pontuação'
```


```python
# A coluna de consentimento pode ser removida, não tem nenhuma informação relevante uma vez que todos os registros são iguais.
df.drop(coluna_1, axis=1, inplace=True)
df.drop(coluna_2, axis=1, inplace=True)
```


```python
# Atribui nomes mais fáceis de usar às colunas da tabela (Use esse método somente se as colunão não forem dinâmicas)
df.rename(columns={df.columns[0]: 'data_hora'}, inplace=True)
df.rename(columns={df.columns[1]: 'faixa_etaria'}, inplace=True)
df.rename(columns={df.columns[2]: 'escolaridade'}, inplace=True)
df.rename(columns={df.columns[3]: 'preoculpacao'}, inplace=True)
df.rename(columns={df.columns[4]: 'acompanha_notícias_vazamentos'}, inplace=True)
df.rename(columns={df.columns[5]: 'impactado_por_vazamentos'}, inplace=True)
df.rename(columns={df.columns[6]: 'confiança_tecnologias_atuais'}, inplace=True)
df.rename(columns={df.columns[7]: 'uso_ia_protecao'}, inplace=True)
df.rename(columns={df.columns[8]: 'opiniao_uso_ia_protecao'}, inplace=True)
df.rename(columns={df.columns[9]: 'uso_novas_tecnologias_controle_dados'}, inplace=True)
df.rename(columns={df.columns[10]: 'importancia_controle_dados_pessoais'}, inplace=True)
df.rename(columns={df.columns[11]: 'comentarios'}, inplace=True)
```


```python
#df.head(3)
```


```python
def exibir_visao_geral_dataframe(df, num_linhas_head=3):
    """
    Exibe uma visão geral do DataFrame, incluindo as primeiras linhas,
    dimensões, informações de tipos de dados e estatísticas descritivas.
    """
    print(f"--- Primeiras {num_linhas_head} linhas do DataFrame ---")
    display(df.head(num_linhas_head)) # Usa display para melhor formatação no Jupyter

    print(f"\n--- Dimensões do DataFrame ---")
    print(f"O DataFrame possui {df.shape[0]} linhas e {df.shape[1]} colunas.")

    print(f"\n--- Informações do DataFrame (df.info()) ---")
    df.info()

    print(f"\n--- Estatísticas Descritivas para Colunas Numéricas ---")
    display(df.describe())

    print(f"\n--- Estatísticas Descritivas para Colunas Categóricas ---")
    display(df.describe(include='object'))
```


```python
exibir_visao_geral_dataframe(df)
```

    --- Primeiras 3 linhas do DataFrame ---



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data_hora</th>
      <th>faixa_etaria</th>
      <th>escolaridade</th>
      <th>preoculpacao</th>
      <th>acompanha_notícias_vazamentos</th>
      <th>impactado_por_vazamentos</th>
      <th>confiança_tecnologias_atuais</th>
      <th>uso_ia_protecao</th>
      <th>opiniao_uso_ia_protecao</th>
      <th>uso_novas_tecnologias_controle_dados</th>
      <th>importancia_controle_dados_pessoais</th>
      <th>comentarios</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2025-07-06 20:35:33.998</td>
      <td>Mais de 60 anos</td>
      <td>Pós-Graduação</td>
      <td>5 (Muito preocupado)</td>
      <td>Diariamente</td>
      <td>Não, que eu saiba</td>
      <td>5 (Confio totalmente)</td>
      <td>Não</td>
      <td>NaN</td>
      <td>Não sei/Não compreendi bem</td>
      <td>1 (Nada importante)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2025-07-06 21:02:45.461</td>
      <td>36 a 45 anos</td>
      <td>Pós-Graduação</td>
      <td>5 (Muito preocupado)</td>
      <td>Frequentemente</td>
      <td>Sim, conheço alguém que foi afetado(a)</td>
      <td>4 (Confio bastante)</td>
      <td>Não</td>
      <td>NaN</td>
      <td>Sim, muito interesse</td>
      <td>5 (Muito importante)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2025-07-07 07:11:55.393</td>
      <td>36 a 45 anos</td>
      <td>Pós-Graduação</td>
      <td>5 (Muito preocupado)</td>
      <td>Diariamente</td>
      <td>Sim, conheço alguém que foi afetado(a)</td>
      <td>2 (Confio pouco)</td>
      <td>Sim</td>
      <td>3 (Moderadamente eficaz)</td>
      <td>Talvez, preciso de mais informações</td>
      <td>5 (Muito importante)</td>
      <td>Uma aplicação de controle total de compartilha...</td>
    </tr>
  </tbody>
</table>
</div>


    
    --- Dimensões do DataFrame ---
    O DataFrame possui 46 linhas e 12 colunas.
    
    --- Informações do DataFrame (df.info()) ---
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 46 entries, 0 to 45
    Data columns (total 12 columns):
     #   Column                                Non-Null Count  Dtype         
    ---  ------                                --------------  -----         
     0   data_hora                             46 non-null     datetime64[ns]
     1   faixa_etaria                          46 non-null     object        
     2   escolaridade                          46 non-null     object        
     3   preoculpacao                          46 non-null     object        
     4   acompanha_notícias_vazamentos         46 non-null     object        
     5   impactado_por_vazamentos              46 non-null     object        
     6   confiança_tecnologias_atuais          46 non-null     object        
     7   uso_ia_protecao                       46 non-null     object        
     8   opiniao_uso_ia_protecao               19 non-null     object        
     9   uso_novas_tecnologias_controle_dados  46 non-null     object        
     10  importancia_controle_dados_pessoais   46 non-null     object        
     11  comentarios                           12 non-null     object        
    dtypes: datetime64[ns](1), object(11)
    memory usage: 4.4+ KB
    
    --- Estatísticas Descritivas para Colunas Numéricas ---



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data_hora</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>46</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2025-07-08 09:58:08.362499840</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2025-07-06 20:35:33.998000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2025-07-08 08:42:25.948499968</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2025-07-08 09:44:29.079000064</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2025-07-08 12:45:02.391250176</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2025-07-10 12:03:13.831000</td>
    </tr>
  </tbody>
</table>
</div>


    
    --- Estatísticas Descritivas para Colunas Categóricas ---



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>faixa_etaria</th>
      <th>escolaridade</th>
      <th>preoculpacao</th>
      <th>acompanha_notícias_vazamentos</th>
      <th>impactado_por_vazamentos</th>
      <th>confiança_tecnologias_atuais</th>
      <th>uso_ia_protecao</th>
      <th>opiniao_uso_ia_protecao</th>
      <th>uso_novas_tecnologias_controle_dados</th>
      <th>importancia_controle_dados_pessoais</th>
      <th>comentarios</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>46</td>
      <td>46</td>
      <td>46</td>
      <td>46</td>
      <td>46</td>
      <td>46</td>
      <td>46</td>
      <td>19</td>
      <td>46</td>
      <td>46</td>
      <td>12</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>5</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>12</td>
    </tr>
    <tr>
      <th>top</th>
      <td>36 a 45 anos</td>
      <td>Pós-Graduação</td>
      <td>5 (Muito preocupado)</td>
      <td>Frequentemente</td>
      <td>Sim, conheço alguém que foi afetado(a)</td>
      <td>3 (Confio razoavelmente)</td>
      <td>Não</td>
      <td>4 (Eficaz)</td>
      <td>Sim, muito interesse</td>
      <td>5 (Muito importante)</td>
      <td>Uma aplicação de controle total de compartilha...</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>19</td>
      <td>31</td>
      <td>21</td>
      <td>29</td>
      <td>19</td>
      <td>25</td>
      <td>27</td>
      <td>9</td>
      <td>20</td>
      <td>35</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


**Tratamento Inicial dos Dados**

A preparação e o tratamento de dados são etapas cruciais em qualquer projeto de análise de dados e Machine Learning, são etapas essenciais para garantir a qualidade, confiabilidade e eficácia das análises e modelos. 

Essa fase inicial corrige inconsistências, imprecisões e gerencia valores ausentes ou duplicados, que, se não tratados, podem distorcer resultados e introduzir vieses. Além disso, muitos algoritmos de aprendizado de máquina são sensíveis a dados brutos; por isso, a preparação, que inclui técnicas como escalonamento, codificação de variáveis categóricas e tratamento de outliers, é fundamental para formatar os dados na escala e no tipo ideais, permitindo que os modelos aprendam com eficácia e gerem previsões precisas. 

Finalmente, com os dados limpos e organizados teremos facilidade na exploração e visualização, o que nos permitirá a identificação de padrões e a obtenção de insights claros. 

Em última análise, devemos buscar como principal foco da preparação dos dados, reduzir erros e vieses para garantirmos que as conclusões e decisões derivadas da análise sejam robustas e válidas, consolidando-se como a base indispensável para o sucesso e a credibilidade de do nosso trabalho de pesquisa.

**Normalizar os dados**

Normalizamos dados principalmente para colocar todas as variáveis em uma escala comum, removendo distorções causadas por grandes diferenças em seus valores e unidades. Isso é crucial para algoritmos de machine learning, especialmente aqueles que dependem de distâncias (como K-NN, SVMs) ou de otimização baseada em gradientes (como redes neurais), pois evita que variáveis com escalas maiores dominem o processo de aprendizado e garante uma convergência mais rápida e estável.


```python
def normalizar_dados_pesquisa(df):
    # Uma cópia do dataset original será aplicada.
    df_normalizado = df.copy()

    # Normalizar colunas com escala numérica explícita (e.g., "5 (Muito preocupado)")
    colunas_escala_numerica = [
        'preoculpacao',
        'confiança_tecnologias_atuais',
        'opiniao_uso_ia_protecao',
        'importancia_controle_dados_pessoais'
    ]
    for coluna in colunas_escala_numerica:
        # Extrai o primeiro dígito da string e converte para numérico
        # Trata NaN, mantendo-os ou convertendo para np.nan se apropriado
        df_normalizado[coluna] = df_normalizado[coluna].astype(str).str[0].replace('n', np.nan).astype(float)

    # Tratamento específico para 'opiniao_uso_ia_protecao':
    # Se 'uso_ia_protecao' for 'Não', a opinião sobre eficácia é irrelevante ou 0.
    # Assumindo que NaN aqui significa que a IA não é usada para proteção, então a opinião é 0.
    # É crucial entender o significado de NaN nesta coluna. Se for "não aplicável", 0 é uma boa escolha.
    df_normalizado['opiniao_uso_ia_protecao'] = df_normalizado['opiniao_uso_ia_protecao'].fillna(0)

    # Mapeamento manual para colunas ordinais com ordem clara
    mapeamento_faixa_etaria = {
        'Até 18 anos': 1,
        '19 a 25 anos': 2,
        '26 a 35 anos': 3,
        '36 a 45 anos': 4,
        '46 a 60 anos': 5,
        'Mais de 60 anos': 6
    }
    df_normalizado['faixa_etaria'] = df_normalizado['faixa_etaria'].map(mapeamento_faixa_etaria)

    mapeamento_escolaridade = {
        'Ensino Fundamental': 1,
        'Ensino Médio': 2,
        'Ensino Superior': 3,
        'Pós-Graduação': 4
    }
    df_normalizado['escolaridade'] = df_normalizado['escolaridade'].map(mapeamento_escolaridade)

    mapeamento_acompanhamento = {
        'Nunca': 1,
        'Raramente': 2,
        'Às vezes': 3,
        'Frequentemente': 4,
        'Diariamente': 5
    }
    df_normalizado['acompanha_notícias_vazamentos'] = df_normalizado['acompanha_notícias_vazamentos'].map(mapeamento_acompanhamento)

    mapeamento_impactado = {
        'Não, que eu saiba': 0,
        'Sim, conheço alguém que foi afetado(a)': 1,
        'Sim, eu fui afetado(a) diretamente': 2
    }
    df_normalizado['impactado_por_vazamentos'] = df_normalizado['impactado_por_vazamentos'].map(mapeamento_impactado)

    mapeamento_interesse_tecnologias = {
        'Não sei/Não compreendi bem': 0,
        'Sim, um pouco de interesse': 1,
        'Talvez, preciso de mais informações': 2,
        'Sim, muito interesse': 3
    }
    df_normalizado['uso_novas_tecnologias_controle_dados'] = df_normalizado['uso_novas_tecnologias_controle_dados'].map(mapeamento_interesse_tecnologias)

    # Mapeamento para coluna binária
    df_normalizado['uso_ia_protecao'] = df_normalizado['uso_ia_protecao'].map({'Não': 0, 'Sim': 1})

    # Converter 'data_hora' para datetime (se necessário para análise de tempo)
    df_normalizado['data_hora'] = pd.to_datetime(df_normalizado['data_hora'])

    return df_normalizado

```


```python
# Aplicar a função de normalização
df_normalizado = normalizar_dados_pesquisa(df)
print(df_normalizado.head())
```

                    data_hora  faixa_etaria  escolaridade  preoculpacao  \
    0 2025-07-06 20:35:33.998             6             4           5.0   
    1 2025-07-06 21:02:45.461             4             4           5.0   
    2 2025-07-07 07:11:55.393             4             4           5.0   
    3 2025-07-07 10:27:58.311             4             4           5.0   
    4 2025-07-07 19:58:10.803             4             3           4.0   
    
       acompanha_notícias_vazamentos  impactado_por_vazamentos  \
    0                              5                         0   
    1                              4                         1   
    2                              5                         1   
    3                              4                         2   
    4                              4                         2   
    
       confiança_tecnologias_atuais  uso_ia_protecao  opiniao_uso_ia_protecao  \
    0                           5.0                0                      0.0   
    1                           4.0                0                      0.0   
    2                           2.0                1                      3.0   
    3                           3.0                0                      0.0   
    4                           3.0                0                      0.0   
    
       uso_novas_tecnologias_controle_dados  importancia_controle_dados_pessoais  \
    0                                     0                                  1.0   
    1                                     3                                  5.0   
    2                                     2                                  5.0   
    3                                     2                                  5.0   
    4                                     2                                  4.0   
    
                                             comentarios  
    0                                                NaN  
    1                                                NaN  
    2  Uma aplicação de controle total de compartilha...  
    3                                                NaN  
    4                                                NaN  



```python
# O atributo comentário tem muito registro nulo, como não vamos usar esse campo diretamente nas análises,
# podemos ajustar futuramente os valores sem removê-los, porque a exclusão pode afetar negativamente a amostra inteira.
df_normalizado.isnull().sum()
```




    data_hora                                0
    faixa_etaria                             0
    escolaridade                             0
    preoculpacao                             0
    acompanha_notícias_vazamentos            0
    impactado_por_vazamentos                 0
    confiança_tecnologias_atuais             0
    uso_ia_protecao                          0
    opiniao_uso_ia_protecao                  0
    uso_novas_tecnologias_controle_dados     0
    importancia_controle_dados_pessoais      0
    comentarios                             34
    dtype: int64



**Exploração Visual dos Dados Normalizados**


```python
# Configurações globais para os gráficos (consistente com o estilo)
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (5, 3)
plt.rcParams['font.size'] = 12
```

**Análise Univariada (Distribuição de Cada Variável)**

A análise univariada (o estudo da distribuição de cada variável individualmente) é um passo inicial e fundamental na exploração de dados. Ela serve para:

1) Obter uma visão rápida da natureza, escala e distribuição de cada variável.
2) Detectar erros de dados, valores anômalos (outliers) e a presença de valores ausentes, o que é crucial para a limpeza dos dados.
3) Medir a tendência central (média, mediana) e a dispersão dos dados (desvio padrão), entendendo quão consistentes ou variados eles são.
4) Deixar as variáveis prontas para análises mais complexas, ajustando-as se necessário para modelos estatísticos.
5) Extrair descobertas preliminares de cada variável antes mesmo de cruzá-las com outras.

Em suma, a análise univariada é a fundação da exploração de dados, garantindo que a qualidade e as características básicas de cada parte do conjunto de dados sejam compreendidas antes de aprofundar a investigação.


```python
def plotar_distribuicao_ordinal(df, coluna, titulo, x_label='Valor da Escala', y_label='Contagem de Respostas', custom_figsize=None):
    """
    Plota a distribuição de uma coluna ordinal numérica usando um countplot,
    com melhorias estéticas e rótulos de eixo X mais descritivos.

    Args:
        df (pd.DataFrame): O DataFrame.
        coluna (str): Nome da coluna a ser plotada.
        titulo (str): Título do gráfico.
        x_label (str): Rótulo do eixo X.
        y_label (str): Rótulo do eixo Y.
        custom_figsize (tuple, optional): Tupla (largura, altura) para o tamanho da figura.
                                          Se None, um tamanho padrão/dinâmico é usado.
    """
    # Definir o tamanho da figura
    if custom_figsize:
        plt.figure(figsize=custom_figsize)
    else:
        num_categorias = df[coluna].nunique()
        largura_sugerida = min(12, max(6, num_categorias * 1.2))
        altura_sugerida = 6
        plt.figure(figsize=(largura_sugerida, altura_sugerida))

    # Configurações estéticas do Seaborn
    sns.set_theme(style="whitegrid", rc={"axes.facecolor": ".9"})
    plt.rcParams['font.size'] = 11

    # Criar o countplot
    # Garante que 'hue' seja a própria coluna para colorir por categoria se necessário,
    # e remove NaNs antes de ordenar para garantir que 'unique()' funcione corretamente.
    ax = sns.countplot(data=df, x=coluna, hue=coluna, palette='viridis',
                       order=sorted(df[coluna].dropna().unique()), # Remova NaNs antes de ordenar
                       legend=False)

    # Adicionar rótulos de valor nas barras (tamanho e posição ajustados)
    for p in ax.patches:
        height = p.get_height()
        if height > 0: # Para evitar rótulos em barras de altura zero
            ax.annotate(f'{int(height)}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center',
                        xytext=(0, 3), # Deslocamento menor (3 pontos acima)
                        textcoords='offset points',
                        fontsize=9, # Tamanho da fonte menor
                        color='black')

    # Configurar Título e Rótulos dos Eixos (tamanho do título reduzido)
    plt.title(f'{titulo}', fontsize=11, fontweight='bold') # Título menor
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)

    # Melhorar os ticks (rótulos dos eixos)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Mapear rótulos do eixo X de volta para os originais ---
    # Define os mapeamentos reversos para cada coluna que foi normalizada
    mapeamentos_reversos = {
        'faixa_etaria': {
            1: 'Até 18 anos', 2: '19 a 25 anos', 3: '26 a 35 anos',
            4: '36 a 45 anos', 5: '46 a 60 anos', 6: 'Mais de 60 anos'
        },
        'escolaridade': {
            1: 'Ensino Fundamental', 2: 'Ensino Médio', 3: 'Ensino Superior', 4: 'Pós-Graduação'
        },
        'acompanha_notícias_vazamentos': { # Nome da coluna corrigido
            1: 'Nunca', 2: 'Raramente', 3: 'Às vezes', 4: 'Frequentemente', 5: 'Diariamente'
        },
        'impactado_por_vazamentos': { # Nome da coluna corrigido
            0: 'Não, que eu saiba', 1: 'Sim, conheço alguém que foi afetado(a)', 2: 'Sim, eu fui afetado(a) diretamente'
        },
        'uso_ia_protecao': {
            0: 'Não', 1: 'Sim'
        },
        'uso_novas_tecnologias_controle_dados': {
            0: 'Não sei/Não compreendi bem', 1: 'Sim, um pouco de interesse',
            2: 'Talvez, preciso de mais informações', 3: 'Sim, muito interesse'
        },
        'preoculpacao': { # Mapeamento para as escalas numéricas 1-5
            1: '1 (Nada preocupado)', 2: '2 (Pouco preocupado)', 3: '3 (Moderadamente preocupado)',
            4: '4 (Preocupado)', 5: '5 (Muito preocupado)'
        },
        'confiança_tecnologias_atuais': { # Mapeamento para as escalas numéricas 1-5
            1: '1 (Não confio)', 2: '2 (Confio pouco)', 3: '3 (Confio razoavelmente)',
            4: '4 (Confio bastante)', 5: '5 (Confio totalmente)'
        },
        'opiniao_uso_ia_protecao': { # Mapeamento para as escalas numéricas 0-5 (0 é para 'Não' uso)
            0: 'Não aplicável (Não usa IA)', 1: '1 (Nada eficaz)', 2: '2 (Pouco eficaz)',
            3: '3 (Moderadamente eficaz)', 4: '4 (Eficaz)', 5: '5 (Muito eficaz)'
        },
        'importancia_controle_dados_pessoais': { # Mapeamento para as escalas numéricas 1-5
            1: '1 (Nada importante)', 2: '2 (Pouco importante)', 3: '3 (Moderadamente importante)',
            4: '4 (Importante)', 5: '5 (Muito importante)'
        }
    }

    if coluna in mapeamentos_reversos:
        mapeamento_reverso = mapeamentos_reversos[coluna]
        # Pega os valores normalizados únicos no eixo X e mapeia para os rótulos originais
        tick_values = sorted(df[coluna].dropna().unique())
        ax.set_xticks(tick_values) # Define a localização dos ticks
        tick_labels = [mapeamento_reverso.get(tick_val, str(tick_val)) # Usa o valor numérico se não houver mapeamento
                       for tick_val in tick_values]
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    else:
        # Para colunas não mapeadas explicitamente, use os valores numéricos como rótulos
        plt.xticks(rotation=45, ha='right')


    # 7. Remover a borda superior e direita
    sns.despine(ax=ax, top=True, right=True)

    # 8. Ajustar layout para evitar sobreposição
    plt.tight_layout()
    plt.show()
```

**Visualizando os dados**


```python
plotar_distribuicao_ordinal(df_normalizado, 'preoculpacao', 'Nível de Preocupação com Vazamentos', x_label='1 (Pouco) - 5 (Muito Preocupado)',custom_figsize=(3, 4))
```


    
![png](PesquisaPublicoGeralPercep%C3%A7%C3%B5es_files/PesquisaPublicoGeralPercep%C3%A7%C3%B5es_31_0.png)
    



```python
plotar_distribuicao_ordinal(df_normalizado, 'confiança_tecnologias_atuais', 'Nível de Confiança nas Organizações', x_label='1 (Não Confio) - 5 (Confio Totalmente)',custom_figsize=(4, 4))
```


    
![png](PesquisaPublicoGeralPercep%C3%A7%C3%B5es_files/PesquisaPublicoGeralPercep%C3%A7%C3%B5es_32_0.png)
    



```python
plotar_distribuicao_ordinal(df_normalizado, 'escolaridade', 'Nível de Escolaridade', x_label='1 (Fundamental) - 4 (Pós-Graduação)',custom_figsize=(3, 4))
```


    
![png](PesquisaPublicoGeralPercep%C3%A7%C3%B5es_files/PesquisaPublicoGeralPercep%C3%A7%C3%B5es_33_0.png)
    



```python
plotar_distribuicao_ordinal(df_normalizado, 'faixa_etaria', 'Faixa Etária', x_label='1 (Até 18) - 6 (Mais de 60)',custom_figsize=(4, 4))
```


    
![png](PesquisaPublicoGeralPercep%C3%A7%C3%B5es_files/PesquisaPublicoGeralPercep%C3%A7%C3%B5es_34_0.png)
    



```python
# Para a coluna binária 'uso_ia_protecao'
plotar_distribuicao_ordinal(df_normalizado, 'uso_ia_protecao', 'Uso de IA para Proteção de Dados', x_label='0 (Não) / 1 (Sim)',custom_figsize=(2, 3))
```


    
![png](PesquisaPublicoGeralPercep%C3%A7%C3%B5es_files/PesquisaPublicoGeralPercep%C3%A7%C3%B5es_35_0.png)
    



```python
# Para 'opiniao_ia_seguranca', onde 0 significa "não aplicável/não usa IA"
plotar_distribuicao_ordinal(df_normalizado, 'opiniao_uso_ia_protecao', 'Opinião sobre Eficácia da IA (0=Não se Aplica)', x_label='0 (N/A) - 5 (Muito Eficaz)',custom_figsize=(4, 4))
```


    
![png](PesquisaPublicoGeralPercep%C3%A7%C3%B5es_files/PesquisaPublicoGeralPercep%C3%A7%C3%B5es_36_0.png)
    


**Análise Bivariada (Relação entre Duas Variáveis)**

A análise bivariada investiga a relação entre duas variáveis simultaneamente, aprofundando o entendimento obtido pela análise univariada. Seu principal objetivo é:

1) Descobrir se e como duas variáveis se conectam (ex: preocupação com vazamento e idade);
2) Expor insights que só aparecem quando as variáveis são vistas em conjunto, como comportamentos específicos de certos grupos;
3) Testar suposições sobre como diferentes fatores se relacionam na pesquisa.
4) Fornecer a base para análises mais complexas e modelos preditivos, ajudando a selecionar variáveis relevantes.



**Nível de Preocupação por Faixa Etária/Escolaridade**


```python
# Mapeamentos para rótulos dos eixos (centralizados para fácil acesso)
# Este dicionário contém todos os mapeamentos reversos das normalizações.
mapeamentos_reversos_globais = {
    'faixa_etaria': {
        1: 'Até 18 anos', 2: '19 a 25 anos', 3: '26 a 35 anos',
        4: '36 a 45 anos', 5: '46 a 60 anos', 6: 'Mais de 60 anos'
    },
    'escolaridade': {
        1: 'Ensino Fundamental', 2: 'Ensino Médio', 3: 'Ensino Superior', 4: 'Pós-Graduação'
    },
    'acompanha_notícias_vazamentos': {
        1: 'Nunca', 2: 'Raramente', 3: 'Às vezes', 4: 'Frequentemente', 5: 'Diariamente'
    },
    'impactado_por_vazamentos': {
        0: 'Não, que eu saiba', 1: 'Sim, conheço alguém que foi afetado(a)', 2: 'Sim, eu fui afetado(a) diretamente'
    },
    'uso_ia_protecao': {
        0: 'Não', 1: 'Sim'
    },
    'uso_novas_tecnologias_controle_dados': {
        0: 'Não sei/Não compreendi bem', 1: 'Sim, um pouco de interesse',
        2: 'Talvez, preciso de mais informações', 3: 'Sim, muito interesse'
    },
    'preoculpacao': {
        1: '1 (Nada preocupado)', 2: '2 (Pouco preocupado)', 3: '3 (Moderadamente preocupado)',
        4: '4 (Preocupado)', 5: '5 (Muito preocupado)'
    },
    'confiança_tecnologias_atuais': {
        1: '1 (Não confio)', 2: '2 (Confio pouco)', 3: '3 (Confio razoavelmente)',
        4: '4 (Confio bastante)', 5: '5 (Confio totalmente)'
    },
    'opiniao_uso_ia_protecao': {
        0: 'Não aplicável (Não usa IA)', 1: '1 (Nada eficaz)', 2: '2 (Pouco eficaz)',
        3: '3 (Moderadamente eficaz)', 4: '4 (Eficaz)', 5: '5 (Muito eficaz)'
    },
    'importancia_controle_dados_pessoais': {
        1: '1 (Nada importante)', 2: '2 (Pouco importante)', 3: '3 (Moderadamente importante)',
        4: '4 (Importante)', 5: '5 (Muito importante)'
    }
}


# --------------------------------------------------------------------------------
# Função plotar_relacao_categoria_ordinal (Boxplot)
# --------------------------------------------------------------------------------
def plotar_relacao_categoria_ordinal(df, x_coluna, y_coluna, titulo, x_label_override=None, y_label_override=None, custom_figsize=None):
    """
    Plota a relação entre uma coluna categórica (x) e uma ordinal numérica (y)
    usando um boxplot para mostrar a distribuição da variável ordinal por categoria,
    com melhorias estéticas e rótulos dinâmicos.

    Args:
        df (pd.DataFrame): O DataFrame.
        x_coluna (str): Nome da coluna categórica (no eixo X).
        y_coluna (str): Nome da coluna ordinal numérica (no eixo Y).
        titulo (str): Título do gráfico.
        x_label_override (str, optional): Rótulo customizado para o eixo X. Se None, será gerado automaticamente.
        y_label_override (str, optional): Rótulo customizado para o eixo Y. Se None, será gerado automaticamente.
        custom_figsize (tuple, optional): Tupla (largura, altura) para o tamanho da figura.
                                          Se None, um tamanho padrão/dinâmico é usado.
    """
    if custom_figsize:
        plt.figure(figsize=custom_figsize)
    else:
        # Ajuste dinâmico do figsize para boxplots
        num_categorias_x = df[x_coluna].nunique()
        largura_sugerida = min(15, max(8, num_categorias_x * 1.5)) # Adapta largura
        altura_sugerida = 7 # Altura um pouco maior para boxplot
        plt.figure(figsize=(largura_sugerida, altura_sugerida))

    sns.set_theme(style="whitegrid", rc={"axes.facecolor": ".9"})
    plt.rcParams['font.size'] = 12

    # A ordem das categorias no eixo X é importante para colunas ordinais normalizadas
    order_x = sorted(df[x_coluna].dropna().unique())

    ax = sns.boxplot(data=df, x=x_coluna, y=y_coluna, hue=x_coluna, palette='coolwarm', legend=False, order=order_x)

    # Melhorias estéticas
    plt.title(f'{titulo}', fontsize=14, fontweight='bold')

    # Rótulo do eixo X: Usa override ou mapeamento reverso
    if x_label_override:
        plt.xlabel(x_label_override, fontsize=12)
    else:
        # Tenta usar o mapeamento reverso para o x_label
        display_name_x = x_coluna.replace('_', ' ').title() # Default name
        if x_coluna in mapeamentos_reversos_globais:
            # Pega o primeiro rótulo original para usar como parte do label, se apropriado
            first_mapped_label = next(iter(mapeamentos_reversos_globais[x_coluna].values()), '').split('(')[0].strip()
            if first_mapped_label: # Se o mapeamento reverso tem um texto útil
                 display_name_x = first_mapped_label + " Categoria" # Generaliza o label
            else: # Se o mapeamento reverso é apenas números, use o nome da coluna formatado
                display_name_x = x_coluna.replace('_', ' ').title()
        plt.xlabel(display_name_x, fontsize=12)


    # Rótulo do eixo Y: Usa override ou mapeamento reverso
    if y_label_override:
        plt.ylabel(y_label_override, fontsize=12)
    else:
        # Tenta usar o mapeamento reverso para o y_label
        display_name_y = y_coluna.replace('_', ' ').title() # Default name
        if y_coluna in mapeamentos_reversos_globais:
            first_mapped_label = next(iter(mapeamentos_reversos_globais[y_coluna].values()), '').split('(')[0].strip()
            if first_mapped_label:
                display_name_y = first_mapped_label + " Escala"
            else:
                display_name_y = y_coluna.replace('_', ' ').title()
        plt.ylabel(display_name_y, fontsize=12)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Mapear rótulos do eixo X de volta para os originais textuais
    if x_coluna in mapeamentos_reversos_globais:
        mapeamento_reverso_x = mapeamentos_reversos_globais[x_coluna]
        # Pega os valores normalizados únicos no eixo X e mapeia para os rótulos originais
        tick_values_x = sorted(df[x_coluna].dropna().unique())
        ax.set_xticks(tick_values_x) # Define a localização dos ticks
        tick_labels_x = [mapeamento_reverso_x.get(val, str(val)) for val in tick_values_x]
        ax.set_xticklabels(tick_labels_x, rotation=45, ha='right')
    else:
        plt.xticks(rotation=45, ha='right') # Rotação padrão se não há mapeamento específico

    # Não mapeamos os ticks do eixo Y em boxplots/violinplots por padrão,
    # pois eles representam uma escala contínua ou ordinal numérica.
    # O 'y_label_override' ou o auto-gerado já é suficiente para indicar a escala.

    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------------
# Função plotar_violino_binario_ordinal (Violinplot)
# --------------------------------------------------------------------------------
def plotar_violino_binario_ordinal(df, bin_coluna, ordinal_coluna, titulo, custom_figsize=None):
    """
    Plota a relação entre uma coluna binária (x) e uma ordinal numérica (y)
    usando um violinplot, com melhorias estéticas e rótulos dinâmicos.

    Args:
        df (pd.DataFrame): O DataFrame.
        bin_coluna (str): Nome da coluna binária (no eixo X).
        ordinal_coluna (str): Nome da coluna ordinal numérica (no eixo Y).
        titulo (str): Título do gráfico.
        custom_figsize (tuple, optional): Tupla (largura, altura) para o tamanho da figura.
                                          Se None, um tamanho padrão/dinâmico é usado.
    """
    if custom_figsize:
        plt.figure(figsize=custom_figsize)
    else:
        # Tamanho fixo, mas ajustável, para violinplot binário
        plt.figure(figsize=(7, 5))

    sns.set_theme(style="whitegrid", rc={"axes.facecolor": ".9"})
    plt.rcParams['font.size'] = 12

    # A ordem das categorias no eixo X é importante para colunas ordinais normalizadas
    order_x = sorted(df[bin_coluna].dropna().unique())

    ax = sns.violinplot(data=df, x=bin_coluna, y=ordinal_coluna, hue=bin_coluna, palette='Pastel1', legend=False, order=order_x)

    # Melhorias estéticas
    plt.title(f'{titulo}', fontsize=14, fontweight='bold')

    # Rótulo do eixo X: Usa mapeamento reverso para a coluna binária
    if bin_coluna in mapeamentos_reversos_globais:
        mapeamento_reverso_bin = mapeamentos_reversos_globais[bin_coluna]
        tick_values_bin = sorted(df[bin_coluna].dropna().unique())
        ax.set_xticks(tick_values_bin)
        bin_labels = [mapeamento_reverso_bin.get(val, str(val)) for val in tick_values_bin]
        plt.xlabel(f'{bin_coluna.replace("_", " ").title()} ({bin_labels[0]} vs {bin_labels[1]})', fontsize=12) # Ajuste o rótulo para refletir binário
        ax.set_xticklabels(bin_labels, fontsize=10)
    else:
        plt.xlabel(bin_coluna.replace('_', ' ').title(), fontsize=12) # Fallback
        plt.xticks(fontsize=10)


    # Rótulo do eixo Y: Usa mapeamento reverso para a coluna ordinal, se disponível
    if ordinal_coluna in mapeamentos_reversos_globais:
        first_mapped_label = next(iter(mapeamentos_reversos_globais[ordinal_coluna].values()), '').split('(')[0].strip()
        if first_mapped_label:
            plt.ylabel(f'{first_mapped_label} Escala', fontsize=12)
        else:
            plt.ylabel(ordinal_coluna.replace('_', ' ').title(), fontsize=12)
    else:
        plt.ylabel(ordinal_coluna.replace('_', ' ').title(), fontsize=12)


    plt.yticks(fontsize=10) # Manter ticks numéricos para o eixo Y em escalas

    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    plt.show()


```


```python
# Nível de Preocupação por Faixa Etária
plotar_relacao_categoria_ordinal(df_normalizado, 'faixa_etaria', 'preoculpacao',
                                 'Nível de Preocupação por Faixa Etária',
                                 x_label_override='Faixa Etária',
                                 y_label_override='Nível de Preocupação (Escala 1-5)')
```


    
![png](PesquisaPublicoGeralPercep%C3%A7%C3%B5es_files/PesquisaPublicoGeralPercep%C3%A7%C3%B5es_41_0.png)
    



```python
# Nível de Confiança por Escolaridade
plotar_relacao_categoria_ordinal(df_normalizado, 'escolaridade', 'confiança_tecnologias_atuais',
                                 'Nível de Confiança por Escolaridade',
                                 x_label_override='Escolaridade',
                                 y_label_override='Nível de Confiança (Escala 1-5)')

```


    
![png](PesquisaPublicoGeralPercep%C3%A7%C3%B5es_files/PesquisaPublicoGeralPercep%C3%A7%C3%B5es_42_0.png)
    



```python
# Confiança nas Tecnologias Atuais vs. Uso de IA para Proteção
plotar_violino_binario_ordinal(df_normalizado, 'uso_ia_protecao', 'confiança_tecnologias_atuais', 
                                 'Confiança nas Tecnologias Atuais vs. Uso de IA para Proteção')
```


    
![png](PesquisaPublicoGeralPercep%C3%A7%C3%B5es_files/PesquisaPublicoGeralPercep%C3%A7%C3%B5es_43_0.png)
    



```python
# Importância do Controle de Dados por Acompanhamento de Vazamentos
plotar_relacao_categoria_ordinal(df_normalizado, 'acompanha_notícias_vazamentos', 'importancia_controle_dados_pessoais',
                                 'Importância do Controle de Dados por Acompanhamento de Vazamentos',
                                 x_label_override='Frequência de Acompanhamento de Notícias de Vazamentos',
                                 y_label_override='Importância do Controle de Dados Pessoais (Escala 1-5)')
```


    
![png](PesquisaPublicoGeralPercep%C3%A7%C3%B5es_files/PesquisaPublicoGeralPercep%C3%A7%C3%B5es_44_0.png)
    


**Matriz de Correlação (para ver relações lineares entre variáveis ordinais)**

A matriz de correlação é uma ferramenta vital para visualizar as relações lineares entre múltiplas variáveis ordinais simultaneamente. Ela é uma tabela que exibe coeficientes de correlação para cada par de variáveis.

Para variáveis ordinais, utilizam-se coeficientes como o de Spearman (ρ) ou Kendall (τ), que medem a relação monotônica (se uma variável tende a aumentar ou diminuir com a outra).

Os coeficientes variam de -1 (correlação negativa perfeita) a +1 (correlação positiva perfeita), com 0 indicando ausência de relação. Valores mais próximos dos extremos indicam relações mais fortes.


```python
def plotar_matriz_correlacao(df, colunas_para_correlacao, titulo='Matriz de Correlação das Variáveis'):
    """
    Plota um heatmap da matriz de correlação para as colunas especificadas.
    Assegura que apenas colunas com dados numéricos sejam consideradas, ignorando NaNs para o cálculo.
    """
    plt.figure(figsize=(10, 8))

    # Seleciona as colunas e remove NaNs para o cálculo da correlação
    # Isso garante que a correlação seja calculada apenas para pares completos de observações.
    df_temp = df[colunas_para_correlacao].dropna()

    if df_temp.empty:
        print("Aviso: Não há dados suficientes (após remover NaNs) para calcular a matriz de correlação para as colunas especificadas.")
        plt.close() # Fecha a figura vazia
        return

    corr_matrix = df_temp.corr()

    # Ajusta o tamanho da anotação dinamicamente com base no número de colunas
    annot_fontsize = max(6, min(10, 30 / len(colunas_para_correlacao)))

    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5,
                annot_kws={"fontsize": annot_fontsize}) # Ajuste dinâmico do tamanho da fonte da anotação
    plt.title(titulo, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10) # Melhora a leitura dos rótulos
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.show()

# --- CORREÇÃO APLICADA AQUI: Nomes das colunas atualizados ---
colunas_numericas_para_correlacao = [
    'faixa_etaria',
    'escolaridade',
    'preoculpacao',
    'acompanha_notícias_vazamentos',      # Corrigido de 'acompanha_vazamentos'
    'impactado_por_vazamentos',          # Corrigido de 'impactado'
    'confiança_tecnologias_atuais',      # Corrigido de 'confiança'
    'uso_ia_protecao',
    'opiniao_uso_ia_protecao',           # Corrigido de 'opiniao_ia_seguranca'
    'uso_novas_tecnologias_controle_dados',
    'importancia_controle_dados_pessoais'
]


plotar_matriz_correlacao(df_normalizado, colunas_numericas_para_correlacao,
                          'Matriz de Correlação entre Variáveis da Pesquisa')
```


    
![png](PesquisaPublicoGeralPercep%C3%A7%C3%B5es_files/PesquisaPublicoGeralPercep%C3%A7%C3%B5es_47_0.png)
    


**Análise Multivariada (Mais de Duas Variáveis)**

Após compreender as características individuais das variáveis (análise univariada) e as relações entre pares de variáveis (análise bivariada), a análise multivariada representa o próximo nível de complexidade e profundidade na exploração de dados. Ela se concentra em examinar as relações entre três ou mais variáveis simultaneamente, buscando padrões, estruturas e interdependências que não seriam evidentes ao analisar os dados em partes menores.

**Pair Plot (Dispersão entre Múltiplas Variáveis)**

O Pair Plot (ou Gráfico de Pares) é uma ferramenta de visualização poderosa e popular na análise multivariada. Embora tecnicamente seja uma coleção de análises bivariadas, sua força reside em apresentar as relações entre todos os pares de variáveis numéricas (e, por extensão, até ordinais, se tratadas como tal) em um único grid visual. Isso permite uma inspeção rápida e eficiente de múltiplos relacionamentos de uma só vez, revelando padrões, correlações e distribuições.


```python
# Customização do gráfico 1
colunas_para_pairplot = [
    'preoculpacao',
    'confiança_tecnologias_atuais',
    'opiniao_uso_ia_protecao',
    'importancia_controle_dados_pessoais',
    'uso_ia_protecao'
]
#Customização do gráfico 2
colunas_para_pairplot_2 = [
    'faixa_etaria',
    'escolaridade',
    'acompanha_notícias_vazamentos'
]

def plotar_pairplot(df, colunas, hue=None, titulo='Pair Plot das Variáveis Selecionadas'):
    """
    Plota um pair plot para as colunas selecionadas, opcionalmente com uma coluna 'hue'.
    Filtra colunas inválidas e verifica se há dados suficientes.
    Otimizado para evitar o UserWarning 'Ignoring `palette` because no `hue` variable has been assigned'.
    """
    # Verificar se todas as colunas existem no DataFrame
    colunas_existentes = [col for col in colunas if col in df.columns]
    colunas_ausentes = [col for col in colunas if col not in df.columns]

    if colunas_ausentes:
        print(f"Aviso: As seguintes colunas não foram encontradas no DataFrame e serão ignoradas: {colunas_ausentes}")

    if not colunas_existentes:
        print("Erro: Nenhuma das colunas especificadas foi encontrada no DataFrame para o pair plot.")
        return

    # Assegurar que as colunas são numéricas e remover NaNs para o pairplot
    df_plot = df[colunas_existentes].apply(pd.to_numeric, errors='coerce')

    # Tratar a coluna 'hue'
    current_hue = hue # Variável para controlar o hue na chamada final do pairplot
    if hue:
        if hue not in df.columns:
            print(f"Aviso: A coluna 'hue' '{hue}' não foi encontrada e será ignorada.")
            current_hue = None
        else:
            # Adicionar a coluna hue ao df_plot (se já não estiver lá)
            if hue not in colunas_existentes:
                df_plot[hue] = df[hue]
            
            # Remover NaNs da coluna hue, e se ela ficar toda NaN, desativar hue
            if df_plot[hue].isnull().all():
                print(f"Aviso: A coluna 'hue' '{hue}' contém apenas NaNs após a limpeza e será ignorada.")
                current_hue = None
            else:
                # Para 'hue', especialmente se for categórico (0/1), converter para object ou category para melhor plotagem
                # e aplicar mapeamento de rótulos se for 'uso_ia_protecao'
                if pd.api.types.is_numeric_dtype(df_plot[hue]) and df_plot[hue].nunique() <= 5:
                    if hue == 'uso_ia_protecao' and hue in mapeamentos_reversos_globais:
                        # Criar uma cópia para não modificar o df original se ele for usado em outro lugar
                        df_plot[hue] = df_plot[hue].map(mapeamentos_reversos_globais[hue]).astype('category')
                    else:
                        df_plot[hue] = df_plot[hue].astype('category')
    
    # Remover linhas onde as colunas numéricas selecionadas para o pairplot são NaN
    # Isso pode ser agressivo, mas garante que sns.pairplot funcione com dados limpos para essas colunas.
    df_plot_final = df_plot.dropna(subset=colunas_existentes) # Apenas dropna nas colunas que serão plotadas

    if df_plot_final.empty:
        print("Aviso: Não há dados suficientes (após remover NaNs) para plotar o pair plot.")
        return

    # Parâmetros para sns.pairplot
    pairplot_kwargs = {
        'diag_kind': 'kde'
    }
    
    # Passar a paleta SOMENTE se houver uma variável hue válida
    if current_hue:
        pairplot_kwargs['hue'] = current_hue
        pairplot_kwargs['palette'] = 'viridis'

    try:
        g = sns.pairplot(df_plot_final, **pairplot_kwargs)
        g.fig.suptitle(titulo, y=1.02) # Ajusta o título geral
        plt.show()
    except ValueError as e:
        print(f"Erro ao gerar pairplot: {e}. Verifique se as colunas selecionadas têm variância suficiente ou dados válidos.")
        plt.close()
```


```python
# Relações entre Preocupação, Confiança, Opinião IA e Importância (por Uso de IA)
plotar_pairplot(df_normalizado, colunas_para_pairplot, hue='uso_ia_protecao',
                 titulo='Relações entre Preocupação, Confiança, Opinião IA e Importância (por Uso de IA)')
```


    
![png](PesquisaPublicoGeralPercep%C3%A7%C3%B5es_files/PesquisaPublicoGeralPercep%C3%A7%C3%B5es_53_0.png)
    



```python
# Relações entre Faixa Etária, Escolaridade e Acompanhamento de Notícias de Vazamentos'
plotar_pairplot(df_normalizado, colunas_para_pairplot_2,
                 titulo='Relações entre Faixa Etária, Escolaridade e Acompanhamento de Notícias de Vazamentos')

```


    
![png](PesquisaPublicoGeralPercep%C3%A7%C3%B5es_files/PesquisaPublicoGeralPercep%C3%A7%C3%B5es_54_0.png)
    


**Análise Temporal (análise desconsiderada porque não é relevante para data_hora)**

Se a coluna data_hora contiver variações significativas ao longo do tempo (por exemplo, meses ou anos de coleta), você pode extrair características temporais. No seu caso, todos os dados são de 6 a 9 de julho de 2025, então a variação temporal é pequena. Mas se fosse uma pesquisa de longa duração, seria útil:

Ao utilizar essas funções e técnicas de visualização, você conseguirá extrair insights valiosos da sua pesquisa qualitativa, apresentando os dados de forma clara e impactante. Lembre-se sempre de interpretar os gráficos à luz dos seus objetivos de pesquisa e das definições dos seus mapeamentos ordinais.

interpretar os gráficos à luz dos objetivos específicos da pesquisa e das definições de quaisquer mapeamentos ordinais.

**Analisando Dados de Opiniões Abertas**

Último atributo é um campo com opiniões abertas (respostas de texto livre) da pesquisa, exige uma abordagem diferente das análises quantitativas. Útilizaremos o método de Processamento de Linguagem Natural (PLN).



```python
# Listar o campo (atributo) comentários
df['comentarios'].head(10)
```




    0                                                  NaN
    1                                                  NaN
    2    Uma aplicação de controle total de compartilha...
    3                                                  NaN
    4                                                  NaN
    5                                                  NaN
    6                                                  NaN
    7                                                  NaN
    8    Temos que levar em consideração o nível de con...
    9                                                  NaN
    Name: comentarios, dtype: object



**Modelagem de Tópicos (LDA) com Gensim**
   
O algoritmo LDA examina os comentários e agrupa palavras que aparecem juntas com frequência, identificando "tópicos" subjacentes.

O LDA revela assuntos maiores e coerentes. Por exemplo, palavras como "segurança", "dados", "privacidade" podem ser agrupadas no Tópico 0: "Preocupação com a Segurança de Dados".

Visualização Interativa (pyLDAvis): O código gera um arquivo HTML (lda_visualization.html) que você pode abrir no seu navegador. Essa ferramenta visual é essencial para explorar os tópicos, ver as palavras que os definem e entender a relação entre eles. Ela te ajuda a nomear e interpretar o que cada tópico realmente representa.



```python
# Isso garante que as colunas 'sentimento' e 'topico_lda' existam,
# mesmo que a análise não seja executada (se df_comentarios_validos for vazio).
df['sentimento'] = None # Ou np.nan, ou 'N/A'
df['topico_lda'] = None # Ou np.nan, ou 'N/A'
# --- FIM DA NOVA ADIÇÃO ---


# Pré-processamento e Limpeza ---

def preprocess_text(text, for_lda=False):
    if pd.isna(text): 
        return "" if not for_lda else []
    text = text.lower()
    text = re.sub(r'[^a-záéíóúçãõâêôü0-9\s]', '', text) 
    tokens = word_tokenize(text, language='portuguese')
    stop_words = set(stopwords.words('portuguese'))
    additional_stop_words = {'muito', 'pouco', 'bom', 'ótimo', 'excelente', 'claro', 'real',
                             'assim', 'gostei', 'gostaria', 'falar', 'poderiam', 'apenas',
                             'coisa', 'fazer', 'mais', 'ser', 'ter', 'seja', 'isso', 'neste',
                             'dessa', 'disso', 'etc', 'ainda', 'sempre', 'cada', 'tudo', 'todos',
                             'alguns', 'algumas', 'onde', 'qual', 'pode', 'pra', 'tão', 'quase',
                             'acho', 'sim', 'nao', 'já'}
    stop_words = stop_words.union(additional_stop_words)

    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return filtered_tokens if for_lda else " ".join(filtered_tokens)

print("Iniciando pré-processamento dos comentários...")
df['comentarios_limpos_str'] = df['comentarios'].apply(lambda x: preprocess_text(x, for_lda=False))
df['comentarios_limpos_tokens'] = df['comentarios'].apply(lambda x: preprocess_text(x, for_lda=True))
print("Pré-processamento concluído.")

df_comentarios_validos = df[df['comentarios_limpos_str'] != ""].copy()

if df_comentarios_validos.empty:
    print("\nNão há comentários válidos para análise após a limpeza e remoção de NaNs. Pulando as análises dependentes.")
    # Se não houver comentários válidos, as colunas 'sentimento' e 'topico_lda' permanecerão None (ou o valor inicial)
    # e o código vai pular para o final.
else:
    print(f"\nTotal de comentários válidos para análise: {len(df_comentarios_validos)}")
    
    # Análise de Frequência de Palavras e Nuvem de Palavras 
    all_words = " ".join(df_comentarios_validos['comentarios_limpos_str'])
    word_counts = Counter(all_words.split())

    print("\n--- Top 10 Palavras Mais Frequentes ---")
    for word, count in word_counts.most_common(10):
        print(f"{word}: {count}")

    print("\nGerando Nuvem de Palavras...")
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          max_words=100, contour_width=3, contour_color='steelblue').generate(all_words)

    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Nuvem de Palavras dos Comentários')
    plt.show()
    print("Nuvem de Palavras gerada.")

    # Análise de Sentimento ---

    print("\nRealizando Análise de Sentimento...")
    from textblob import TextBlob 
    def get_sentiment(text):
        if not text:
            return None
        try:
            analysis = TextBlob(text)
            if analysis.sentiment.polarity > 0:
                return 'Positivo'
            elif analysis.sentiment.polarity < 0:
                return 'Negativo'
            else:
                return 'Neutro'
        except Exception as e:
            return 'Neutro'

    # A coluna 'sentimento' é criada aqui para df_comentarios_validos
    df_comentarios_validos['sentimento'] = df_comentarios_validos['comentarios_limpos_str'].apply(get_sentiment)
    # E atualizada no df original, garantindo que a coluna exista
    df['sentimento'] = df_comentarios_validos['sentimento'] # Atribui de volta ao df original

    print("\n--- Distribuição de Sentimento ---")
    sentiment_counts = df_comentarios_validos['sentimento'].value_counts(dropna=False)
    print(sentiment_counts)

    plt.figure(figsize=(8, 5))
    sentiment_counts.plot(kind='bar', color=['skyblue', 'lightcoral', 'lightgray'])
    plt.title('Distribuição do Sentimento nos Comentários')
    plt.xlabel('Sentimento')
    plt.ylabel('Número de Comentários')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
```

    Iniciando pré-processamento dos comentários...
    Pré-processamento concluído.
    
    Total de comentários válidos para análise: 12
    
    --- Top 10 Palavras Mais Frequentes ---
    dados: 12
    forma: 4
    segurança: 4
    proteção: 3
    quesito: 3
    código: 3
    precisa: 2
    gente: 2
    sabe: 2
    qualquer: 2
    
    Gerando Nuvem de Palavras...



    
![png](PesquisaPublicoGeralPercep%C3%A7%C3%B5es_files/PesquisaPublicoGeralPercep%C3%A7%C3%B5es_62_1.png)
    


    Nuvem de Palavras gerada.
    
    Realizando Análise de Sentimento...
    
    --- Distribuição de Sentimento ---
    sentimento
    Neutro      10
    Positivo     2
    Name: count, dtype: int64



    
![png](PesquisaPublicoGeralPercep%C3%A7%C3%B5es_files/PesquisaPublicoGeralPercep%C3%A7%C3%B5es_62_3.png)
    



```python
# Análise de N-Gramas (Bi-gramas e Tri-gramas) ---
print("\n--- Análise de N-Gramas (Bi-gramas e Tri-gramas) ---")

all_tokens = [token for sublist in df_comentarios_validos['comentarios_limpos_tokens'] for token in sublist]

bigrams = list(nltk.bigrams(all_tokens))
bigram_freq = Counter(bigrams)
print("\nTop 10 Bi-gramas Mais Frequentes:")
for bigram, count in bigram_freq.most_common(10):
    print(f"{' '.join(bigram)}: {count}")

trigrams = list(nltk.trigrams(all_tokens))
trigram_freq = Counter(trigrams)
print("\nTop 10 Tri-gramas Mais Frequentes:")
for trigram, count in trigram_freq.most_common(10):
    print(f"{' '.join(trigram)}: {count}")

# Modelagem de Tópicos (LDA) ---
print("\n--- Modelagem de Tópicos (LDA) ---")

processed_docs_for_gensim = df_comentarios_validos['comentarios_limpos_tokens'].tolist()

if not processed_docs_for_gensim:
    print("Aviso: Não há documentos válidos para a Modelagem de Tópicos após o pré-processamento. Pulando LDA.")
else:
    dictionary = corpora.Dictionary(processed_docs_for_gensim)
    dictionary.filter_extremes(no_below=2, no_above=0.5)

    if not dictionary:
        print("Aviso: O dicionário está vazio após a filtragem. Não há termos suficientes para computar LDA. Pulando LDA.")
    else:
        corpus = [dictionary.doc2bow(doc) for doc in processed_docs_for_gensim]
        corpus = [doc for doc in corpus if doc]
        
        if not corpus:
            print("Aviso: O corpus está vazio após a filtragem de termos. Não há dados suficientes para computar LDA. Pulando LDA.")
        else:
            num_topics = 3
            print(f"\nTreinando modelo LDA com {num_topics} tópicos...")
            lda_model = models.LdaMulticore(corpus=corpus,
                                            id2word=dictionary,
                                            num_topics=num_topics,
                                            random_state=100,
                                            chunksize=100,
                                            passes=10,
                                            per_word_topics=True)
            print("Modelo LDA treinado.")

            print("\n--- Tópicos Identificados pelo LDA ---")
            for idx, topic in lda_model.print_topics(-1):
                print(f"Tópico #{idx}: {topic}")

            print("\nGerando visualização interativa dos tópicos (pyLDAvis)...")
            vis = gensim_models.prepare(lda_model, corpus, dictionary, sort_topics=False)
            pyLDAvis.save_html(vis, 'lda_visualization.html')
            print("Visualização LDA salva em 'lda_visualization.html'. Abra este arquivo no seu navegador.")

            # O trecho abaixo agora atribui os tópicos ao df_comentarios_validos
            # e depois copia esses resultados para as colunas pré-inicializadas no df original.
            topic_assignments = [None] * len(df_comentarios_validos)
            for i, doc_bow in enumerate(corpus):
                if doc_bow:
                    topics_for_doc = lda_model[doc_bow][0]
                    if topics_for_doc:
                        dominant_topic = max(topics_for_doc, key=lambda x: x[1])
                        topic_assignments[i] = dominant_topic[0]
            
            df_comentarios_validos['topico_lda'] = topic_assignments
            
            # Atribui os resultados de volta ao DataFrame original
            # Isso é importante para que as colunas existam no 'df' mesmo se o df_comentarios_validos for vazio
            df['topico_lda'] = df_comentarios_validos['topico_lda']


# Agora as colunas 'sentimento' e 'topico_lda' sempre existirão no 'df',
# mesmo que estejam cheias de 'None's se a análise não pôde ser feita.
print(df[['comentarios', 'sentimento', 'topico_lda']].head(10))

print("\nAnálise completa da coluna 'comentarios' concluída.")
print("\nVerifique o arquivo 'lda_visualization.html' no diretório do seu script para a visualização dos tópicos.")
```

    
    --- Análise de N-Gramas (Bi-gramas e Tri-gramas) ---
    
    Top 10 Bi-gramas Mais Frequentes:
    proteção dados: 3
    dados pessoais: 2
    aplicação controle: 1
    controle total: 1
    total compartilhamento: 1
    compartilhamento dados: 1
    dados parte: 1
    parte titular: 1
    titular inviabilizaria: 1
    inviabilizaria negócios: 1
    
    Top 10 Tri-gramas Mais Frequentes:
    aplicação controle total: 1
    controle total compartilhamento: 1
    total compartilhamento dados: 1
    compartilhamento dados parte: 1
    dados parte titular: 1
    parte titular inviabilizaria: 1
    titular inviabilizaria negócios: 1
    inviabilizaria negócios precisa: 1
    negócios precisa aderente: 1
    precisa aderente leis: 1
    
    --- Modelagem de Tópicos (LDA) ---
    
    Treinando modelo LDA com 3 tópicos...
    Modelo LDA treinado.
    
    --- Tópicos Identificados pelo LDA ---
    Tópico #0: 0.251*"segurança" + 0.155*"pessoas" + 0.122*"processos" + 0.122*"falta" + 0.026*"momento" + 0.026*"precisa" + 0.026*"sistema" + 0.023*"tecnologias" + 0.023*"nesse" + 0.023*"empresas"
    Tópico #1: 0.153*"forma" + 0.117*"proteção" + 0.082*"vez" + 0.082*"empresas" + 0.082*"nesse" + 0.082*"tecnologias" + 0.047*"sabe" + 0.047*"momento" + 0.047*"meio" + 0.047*"alguma"
    Tópico #2: 0.125*"seguro" + 0.125*"qualquer" + 0.125*"gente" + 0.072*"sistema" + 0.072*"precisa" + 0.072*"alguma" + 0.072*"meio" + 0.072*"sabe" + 0.069*"momento" + 0.035*"segurança"
    
    Gerando visualização interativa dos tópicos (pyLDAvis)...
    Visualização LDA salva em 'lda_visualization.html'. Abra este arquivo no seu navegador.
                                             comentarios sentimento  topico_lda
    0                                                NaN        NaN         NaN
    1                                                NaN        NaN         NaN
    2  Uma aplicação de controle total de compartilha...   Positivo         1.0
    3                                                NaN        NaN         NaN
    4                                                NaN        NaN         NaN
    5                                                NaN        NaN         NaN
    6                                                NaN        NaN         NaN
    7                                                NaN        NaN         NaN
    8  Temos que levar em consideração o nível de con...     Neutro         2.0
    9                                                NaN        NaN         NaN
    
    Análise completa da coluna 'comentarios' concluída.
    
    Verifique o arquivo 'lda_visualization.html' no diretório do seu script para a visualização dos tópicos.



```python
from IPython.display import HTML
file_path = 'lda_visualization.html'
display(HTML(html_content))
```



<link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v1.0.0.css">


<div id="ldavis_el55871404070648411209706131433" style="background-color:white;"></div>
<script type="text/javascript">

var ldavis_el55871404070648411209706131433_data = {"mdsDat": {"x": [-0.13049496870386293, 0.06761034363236677, 0.06288462507149614], "y": [-0.0022622452165421414, -0.09257260147006538, 0.09483484668660754], "topics": [1, 2, 3], "cluster": [1, 1, 1], "Freq": [23.370202042260384, 46.43015250583597, 30.199645451903645]}, "tinfo": {"Term": ["seguran\u00e7a", "pessoas", "seguro", "qualquer", "gente", "processos", "falta", "forma", "prote\u00e7\u00e3o", "vez", "alguma", "meio", "empresas", "nesse", "tecnologias", "sistema", "sabe", "precisa", "momento", "pessoas", "seguran\u00e7a", "processos", "falta", "momento", "precisa", "sistema", "tecnologias", "nesse", "empresas", "gente", "qualquer", "seguro", "sabe", "alguma", "meio", "vez", "prote\u00e7\u00e3o", "forma", "forma", "prote\u00e7\u00e3o", "vez", "empresas", "nesse", "tecnologias", "sabe", "momento", "meio", "alguma", "precisa", "sistema", "falta", "processos", "seguran\u00e7a", "pessoas", "gente", "qualquer", "seguro", "seguro", "qualquer", "gente", "sistema", "precisa", "alguma", "meio", "sabe", "momento", "seguran\u00e7a", "pessoas", "processos", "falta", "tecnologias", "nesse", "vez", "empresas", "prote\u00e7\u00e3o", "forma"], "Freq": [3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.554381395663571, 2.5228156529388563, 1.2235045602882042, 1.223475509845077, 0.2613158300781121, 0.2589977469738792, 0.2571177041209766, 0.2335258617189892, 0.23339683806919773, 0.23302240181641534, 0.231436064184589, 0.23132325038257343, 0.2307710299101886, 0.22429515828756968, 0.22358783989524111, 0.22354948657180826, 0.22370462192789006, 0.23019080720654192, 0.22877613298574673, 3.0450610038863664, 2.3384807069196336, 1.6414613537281042, 1.632046483026215, 1.6311431207306597, 1.6308750721786112, 0.940025818828755, 0.9335983804770686, 0.9335425990081351, 0.9334026990840498, 0.8953325902898498, 0.8945644422749894, 0.5850000113456997, 0.5849615593197814, 0.40427604734501454, 0.2353846716603809, 0.23538822308056967, 0.23522177117727208, 0.23519943999587573, 1.6205150002750472, 1.619914072964029, 1.6196306860546787, 0.9392790900174419, 0.9365478749051025, 0.9359109558118108, 0.9358124621710739, 0.9286040434977332, 0.89618696116982, 0.44839418612067644, 0.23352133509561865, 0.23366007761115568, 0.23365245838305349, 0.23409693754370697, 0.23396661246112094, 0.23389327436713414, 0.23346350152611917, 0.2346300658186288, 0.23416766333091588], "Total": [3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0232874024195704, 3.3754858864045474, 2.042126197219141, 2.04212797957383, 2.0911011717250005, 2.0908782121688314, 2.090961236413408, 2.0984978714413076, 2.0985065712609785, 2.0985323863687495, 2.0864549733198374, 2.0864590945238746, 2.0864854701811115, 2.092925020614058, 2.0929014947911018, 2.0929045477510173, 2.0990592500231284, 2.8033015799448044, 3.508004800203029, 3.508004800203029, 2.8033015799448044, 2.0990592500231284, 2.0985323863687495, 2.0985065712609785, 2.0984978714413076, 2.092925020614058, 2.0911011717250005, 2.0929045477510173, 2.0929014947911018, 2.0908782121688314, 2.090961236413408, 2.04212797957383, 2.042126197219141, 3.3754858864045474, 2.0232874024195704, 2.0864549733198374, 2.0864590945238746, 2.0864854701811115, 2.0864854701811115, 2.0864590945238746, 2.0864549733198374, 2.090961236413408, 2.0908782121688314, 2.0929014947911018, 2.0929045477510173, 2.092925020614058, 2.0911011717250005, 3.3754858864045474, 2.0232874024195704, 2.042126197219141, 2.04212797957383, 2.0984978714413076, 2.0985065712609785, 2.0990592500231284, 2.0985323863687495, 2.8033015799448044, 3.508004800203029], "Category": ["Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3"], "logprob": [19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -1.8664, -1.3821, -2.1058, -2.1058, -3.6495, -3.6584, -3.6657, -3.762, -3.7625, -3.7641, -3.7709, -3.7714, -3.7738, -3.8023, -3.8054, -3.8056, -3.8049, -3.7763, -3.7825, -1.8805, -2.1445, -2.4984, -2.5041, -2.5047, -2.5049, -3.0558, -3.0627, -3.0627, -3.0629, -3.1045, -3.1054, -3.5301, -3.5302, -3.8996, -4.4405, -4.4405, -4.4412, -4.4413, -2.0811, -2.0815, -2.0817, -2.6265, -2.6294, -2.6301, -2.6302, -2.6379, -2.6735, -3.3659, -4.0183, -4.0177, -4.0178, -4.0159, -4.0164, -4.0168, -4.0186, -4.0136, -4.0156], "loglift": [19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.1901, 1.1625, 0.9414, 0.9414, -0.626, -0.6348, -0.6421, -0.742, -0.7425, -0.7442, -0.7452, -0.7457, -0.7481, -0.7796, -0.7828, -0.783, -0.7852, -1.0459, -1.2764, 0.6257, 0.5859, 0.5213, 0.5158, 0.5153, 0.5151, -0.0332, -0.0392, -0.0401, -0.0402, -0.0809, -0.0818, -0.4829, -0.483, -1.355, -1.384, -1.4148, -1.4155, -1.4156, 0.9446, 0.9442, 0.9441, 0.3971, 0.3942, 0.3926, 0.3924, 0.3847, 0.35, -0.8213, -0.9619, -0.9705, -0.9706, -0.9959, -0.9965, -0.997, -0.9986, -1.2832, -1.5094]}, "token.table": {"Topic": [2, 3, 2, 1, 2, 2, 3, 2, 3, 2, 3, 2, 1, 2, 3, 1, 2, 2, 3, 2, 3, 1, 3, 2, 3, 2, 2], "Freq": [0.47780557397891904, 0.47780557397891904, 0.9530470022722654, 0.48968527438162285, 0.48968527438162285, 0.855186971188401, 0.958563700427105, 0.4778048769947845, 0.4778048769947845, 0.4782169382914532, 0.4782169382914532, 0.9530587263294646, 0.9884903141334632, 0.47826793267060613, 0.47826793267060613, 0.48968570177579956, 0.48968570177579956, 0.7134444664492284, 0.9585618070582859, 0.47780020313704447, 0.47780020313704447, 0.8887609372277654, 0.9585496896972858, 0.4782489424410774, 0.4782489424410774, 0.9530626774600174, 0.9528077875733918], "Term": ["alguma", "alguma", "empresas", "falta", "falta", "forma", "gente", "meio", "meio", "momento", "momento", "nesse", "pessoas", "precisa", "precisa", "processos", "processos", "prote\u00e7\u00e3o", "qualquer", "sabe", "sabe", "seguran\u00e7a", "seguro", "sistema", "sistema", "tecnologias", "vez"]}, "R": 19, "lambda.step": 0.01, "plot.opts": {"xlab": "PC1", "ylab": "PC2"}, "topic.order": [1, 2, 3]};

function LDAvis_load_lib(url, callback){
  var s = document.createElement('script');
  s.src = url;
  s.async = true;
  s.onreadystatechange = s.onload = callback;
  s.onerror = function(){console.warn("failed to load library " + url);};
  document.getElementsByTagName("head")[0].appendChild(s);
}

if(typeof(LDAvis) !== "undefined"){
   // already loaded: just create the visualization
   !function(LDAvis){
       new LDAvis("#" + "ldavis_el55871404070648411209706131433", ldavis_el55871404070648411209706131433_data);
   }(LDAvis);
}else if(typeof define === "function" && define.amd){
   // require.js is available: use it to load d3/LDAvis
   require.config({paths: {d3: "https://d3js.org/d3.v5"}});
   require(["d3"], function(d3){
      window.d3 = d3;
      LDAvis_load_lib("https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v3.0.0.js", function(){
        new LDAvis("#" + "ldavis_el55871404070648411209706131433", ldavis_el55871404070648411209706131433_data);
      });
    });
}else{
    // require.js not available: dynamically load d3 & LDAvis
    LDAvis_load_lib("https://d3js.org/d3.v5.js", function(){
         LDAvis_load_lib("https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v3.0.0.js", function(){
                 new LDAvis("#" + "ldavis_el55871404070648411209706131433", ldavis_el55871404070648411209706131433_data);
            })
         });
}
</script>


### Conclusão ###

A escolha do pipeline de processamento de dados, fundamentado no ambiente Jupyter Notebook com Python e suas bibliotecas, baseou-se em premissas essenciais que garantem a eficácia e a viabilidade do projeto. Optou-se por essa abordagem pela flexibilidade e fácil customização dos códigos, permitindo adaptar o tratamento dos dados às particularidades da pesquisa. A natureza sem custos dessas ferramentas de código aberto democratiza o acesso a recursos de análise avançada. A reprodutibilidade do processo através do Jupyter Notebook assegura que cada etapa da análise seja transparente e replicável, fortalecendo a credibilidade dos resultados.

**Próximos Passos:**

O foco será correlacionar esses achados diretamente com os objetivos do TCC, buscando compreender a percepção pública sobre os riscos de vazamento de dados, o nível de preocupação dos participantes e a eficácia percebida das soluções de segurança digital, com especial atenção à Identidade Auto Soberana (SSI). Esses resultados serão a base da seção de Resultados Preliminares do TCC, oferecendo diretrizes valiosas para o uso da Inteligência Artificial (IA) e da SSI no combate aos vazamentos de dados.


```python

```
