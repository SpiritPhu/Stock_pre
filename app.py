import pandas as pd
from vnstock import *
import streamlit as st
from draw_candlestick_complex import get_candlestick_plot
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit_extras.no_default_selectbox import selectbox

##################################################
# Contents of ~/my_app/main_page.py
import streamlit as st
st.markdown("# Thông tin chứng khoán và Dự báo 💰")
st.sidebar.markdown("# Thông tin chứng khoán và Dự báo 💰")

company_listed = pd.read_csv('/root/phuuu/own-professional-stock/listing_companies_enhanced-2023.csv')
# company_listed = pd.read_csv('/Users/Phu/stock_platform/Own_stock/listing_companies_enhanced-2023.csv')
company_ticker = company_listed['ticker']
company_name = company_listed['organShortName']


# Sidebar options
list_remove = ['VVS', 'XDC', 'HSV']
ticker = st.sidebar.selectbox(
    'Chọn Mã Cổ phiếu', 
    options = [ c for c in company_ticker if c not in list_remove]
)

days_to_plot = st.sidebar.slider(
    'Số ngày hiển thị',
    min_value = 1,
    max_value = 500,
    value = 300,
)
ma1 = st.sidebar.number_input(
    'Moving Average #1 Length',
    value = 10,
    min_value = 1,
    max_value = 120,
    step = 1,
)
ma2 = st.sidebar.number_input(
    'Moving Average #2 Length',
    value = 20,
    min_value = 1,
    max_value = 120,
    step = 1,
)

# Get the dataframe and add the moving averages
from vnstock import *
from datetime import datetime, time, timedelta, timezone

yesterday = datetime.now() - timedelta(1)
yesterday = datetime.strftime(yesterday, '%Y-%m-%d')
today = datetime.now()
today = datetime.strftime(today, '%Y-%m-%d')

df = stock_historical_data(symbol= f'{ticker}',
                               start_date="2010-01-01",
                                end_date = today)

# Rename columns
df.rename(columns={"Open":"open","High":"high",\
                   "Low":"low","Close":"close", 'Volume' : 'volume'}, inplace= True)

print(df)
df.dropna(inplace=True)
df.isna().any()
df['date'] = pd.to_datetime(df['time'], format = '%Y-%m-%d')
df[f'{ma1}_ma'] = df['close'].rolling(ma1).mean()
df[f'{ma2}_ma'] = df['close'].rolling(ma2).mean()

df_plot = df[-days_to_plot:]
st.title("Tên Công ty:")
st.subheader(company_listed[company_listed['ticker'] == ticker]['organName'])

######metric box####
df_metric_today = df.tail(1)['close'] 
df_metric_yestoday = df.shift(1).tail(1)['close']
st.metric(f'🚀 Ngày cập nhật: {df.tail(1).time} (Tăng/ Giảm so ngày hôm qua):', int(df_metric_today), int(df_metric_today) - int(df_metric_yestoday))
col1, col2, col3 = st.columns(3)
# col1.metric("Temperature", "70 °F", "1.2 °F")
# col2.metric("Wind", "9 mph", "-8%")
# col3.metric("Humidity", "86%", "4%")

###Visualize thông tin
#Thông tin cơ bản
df_tt = price_board(f'{ticker}')
df_ttcb = df_tt[['P/E', 'P/B', 'ROE', 'TCBS định giá', 'Đỉnh 1M', 'Đỉnh 3M', 'Đỉnh 1Y', 'Đáy 1M', 'Đáy 3M', 'Đáy 1Y','% thay đổi giá 3D',
       '% thay đổi giá 1M', '% thay đổi giá 3M', '% thay đổi giá 1Y']]
df_ttcb = df_ttcb.rename(columns = {'TCBS định giá' : 'CTCK định giá'})

#Thông tin kỹ thuật
df_ttkt = df_tt[['RSI', 'MACD Signal', 'Tín hiệu KT', 'Tín hiệu TB động', 'MA20', 'MA50', 'Đỉnh 3M', 'Đỉnh 1Y',
       'Đáy 1M', 'Đáy 3M', 'Đáy 1Y', '%Đỉnh 1Y', '%Đáy 1Y']]

#Phân tích ngành
df_ptn = industry_analysis(f'{ticker}')
# print(df_ptn)
# df_ptn = df_ptn[['ticker', 'priceToBook', 'income5year', 'sale5year', 'income1quarter', 'sale1quarter', 'nextIncome', 'nextSale']]
df_ptn

st.subheader("Thông tin PT cơ bản:")
st.dataframe(df_ttcb.style.highlight_max(axis=0))
st.subheader("Thông tin PTKT cơ bản:")
st.dataframe(df_ttkt.style.highlight_max(axis=0))


# Display the plotly chart on the dashboard
fig1 = get_candlestick_plot(df_plot, ma1, ma2, ticker)
fig1.update_layout(autosize=False,
    width=18000,
    height=500,)
st.plotly_chart(
    fig1,
    theme="streamlit",
    use_container_width = True,
)

st.subheader("Phân tích ngành:")
# st.dataframe(df_ptn.style.highlight_max(axis=0))
st.dataframe(df_ptn.style)


###
from model_svr import svr_model
from model_rf import rf_model
from model_knn import knn_model
from model_lstm import lstm_model
import time as t

st.subheader('Nhấn vào đây để thực hiện dự báo xu hướng mã cổ phiếu cần xem, chờ khoảng 1 phút để mô hình thực hiện các bước phân tích, huấn luyện')
predict = st.button('Dự báo xu hướng 15 ngày tới')
if predict:
    st.header('Model is loading:')
    st.markdown("![Alt Text](https://i.gifer.com/YVPG.gif)")
    ###Make a timer

    ###
    svrdf = svr_model(df)
    rfdf = rf_model(df)
    knndf = knn_model(df)
    lstmdf = lstm_model(df)

    finaldf = pd.DataFrame({
        # 'date' : full_dates,
        'svr': svrdf,
        'rf': rfdf,
        'lstm': lstmdf,
        'KNN': knndf,
    })
    finaldf['predict_avg'] = finaldf.mean(axis=1)

    next_15days_df = []
    for x in range(1, 16):
        next_15days_ = datetime.now() + timedelta(x)
        next_15days_df.append(datetime.strftime(next_15days_, '%Y-%m-%d'))
    next_15days = pd.DataFrame(next_15days_df, columns=['dates'])
    full_dates = df['date']._append(next_15days['dates'])
    full_dates = pd.DataFrame(full_dates)
    full_dates = full_dates.rename(columns={'0': 'date'})
    full_dates = full_dates.reset_index(drop=True)
    full_dates = full_dates.reset_index()

    finaldf = finaldf.reset_index()
    final_df = pd.merge(full_dates, finaldf, how = 'left', on = 'index')
    final_df = final_df.rename(columns={0: 'date'})

    #####Echarts#####
    # Conclusion Chart

    names = cycle(['SVR', 'rf', 'LSTM', 'KNN', 'Giá dự báo'])

    fig = px.line(final_df[-200:], x=final_df.date[-200:], y=[final_df['svr'][-200:], final_df['rf'][-200:],
                                                              final_df['lstm'][-200:],
                                                              final_df['KNN'][-200:],
                                                              final_df['predict_avg'][-200:]],
                  labels={'x': 'Thời gian', 'value': 'Giá đóng cửa'})
    fig.update_layout(title_text='Dự báo xu hướng giá chứng khóan 15 ngày tới', font_size=15, font_color='black',
                      legend_title_text='Algorithms')
    fig.for_each_trace(lambda t: t.update(name=next(names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig, use_container_width=True)


# Contents of ~/my_app/pages/page_2.py
import streamlit as st

st.markdown("# Thư viện tra cứu - Tàng Kinh Các 📚️")
st.sidebar.markdown("# Thư viện tra cứu 📚 ")

# from openai import OpenAI
# import streamlit as st
# from streamlit_chat import message

# client = OpenAI()

# openai.api_key = st.secrets['api_secret']

# #create a function which will generate the calls from the api

# def generate_response(prompt):
#     completions = openai.Completion.create(
#         engine = "gpt-3.5-turbo",
#         prompt = prompt,
#         max_tokens = 1024,
#         n = 1,
#         stop = None,
#         temperature = 0.5,

#     )
#     message = completions.choices[0].text
#     return message

# #st.image('/root/phuuu/Chatbot-Openai/logofis.png', use_column_width=True)
# # st.markdown("![Alt Text](https://media.giphy.com/media/2A1FfWjPqZdpXYb9Ur/giphy.gif)")

# if 'generated' not in st.session_state:
#     st.session_state['generated'] = []

# if 'past' not in st.session_state:
#     st.session_state['past'] = []

# def get_text():
#     input_text = st.text_input("Nhập thông tin cần tìm hiểu ở đây (sau đó nhấn Enter): ", "Xin chào Phulh's Bot, Mô hình dự báo LSTM bạn sử dụng là gì?", key = "input")
#     return input_text

# user_input = get_text()

# if user_input:
#     output = generate_response(user_input)
#     #store the output
#     st.session_state.past.append(user_input)
#     st.session_state.generated.append(output)

# if st.session_state['generated']:

#     for i in range(len(st.session_state['generated']) - 1, -1, -1):
#         message(st.session_state["generated"][i], key=str(i))
#         message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

