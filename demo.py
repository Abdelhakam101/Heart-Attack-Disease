import streamlit as st
import numpy as np 
import pandas as pd 
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.express as px 
import plotly.graph_objects as go 
import matplotlib.pyplot as plt 
from PIL import Image
import os
import cProfile
pd.set_option("display.max_columns" , None)
def main():


    st.set_page_config(page_title="Heart Attack Disease",layout="wide")

    # Using object notation
    add_selectbox = st.sidebar.selectbox(
        "Pages",
        ("Info","Describtive Stats", "EDA","Conclusion")
    )
    #####################################################################################################################################
    df=pd.read_csv("EDA.csv")
    image = Image.open(os.path.join(os.getcwd(),"heart-and-ekg.jpg"))
    st.image(image, caption=' Heart Attack Disease')
    if add_selectbox =="Info":
        st.title("EDA for Heart Attack Disease , By [Abdelhakam Ashraf](https://www.linkedin.com/in/abdelhakam-ashraf-056393258/)")

                        
        st.header("About :")
        st.subheader("Key Indicators of Heart Disease")
        st.subheader("2022 annual CDC survey data of 400k+ adults related to their health status")
        st.write('- __State__: The state in which the individual resides.')
        st.write('- __Sex__: The gender of the individual.')
        st.write('- __GeneralHealth__: A subjective measure of the individual overall health.')
        st.write('- __PhysicalHealthDays__: The number of days in which the individual experienced physical health issues.')
        st.write('- __MentalHealthDays__: The number of days in which the individual experienced mental health issues.')
        st.write('- __PhysicalActivities__: Information about the individual engagement in physical activities')
        st.write('- __LastCheckupTime__: Time since the individual last health checkup.')
        st.write('- __SleepHours__: The average number of hours of sleep the individual gets.')
        st.write('- __RemovedTeeth__: Whether the individual has had teeth removed.')
        st.write('- __HadHeartAttack__: Whether the individual has had a heart attack.')
        st.write('- __HadAngina__: Whether the individual has experienced angina (chest pain or discomfort).')
        st.write('- __HadStroke__: Whether the individual has had a stroke.')
        st.write('- __HadAsthma__: Whether the individual has had asthma.')
        st.write('- __HadSkinCancer__: Whether the individual has had skin cancer.')
        st.write('- __HadCOPD__: Whether the individual has chronic obstructive pulmonary disease (COPD).')
        st.write('- __HadDepressiveDisorder__: Whether the individual has had a depressive disorder.')
        st.write('- __HadKidneyDisease__: Whether the individual has had kidney disease.')
        st.write('-  __HadArthritis__: Whether the individual has had arthritis.')
        st.write('- __HadDiabetes__: Whether the individual has diabetes.')
        st.write('- __DeafOrHardOfHearing__: Whether the individual is deaf or hard of hearing.')
        st.write('- __BlindOrVisionDifficulty__: Whether the individual has blindness or difficulty with vision.')
        st.write('- __DifficultyConcentrating__: Whether the individual experiences difficulty concentrating.')
        st.write('- __DifficultyWalking__: Whether the individual experiences difficulty walking.')
        st.write('- __DifficultyDressingBathing__: Whether the individual has difficulty with dressing or bathing.')
        st.write('- __DifficultyErrands__: Whether the individual has difficulty running errands.')
        st.write('- __SmokerStatus__: The smoking status of the individual.')
        st.write('- __ECigaretteUsage__: Whether the individual uses e-cigarettes.')
        st.write('- __ChestScan__: Whether the individual has had a chest scan.')
        st.write('- __RaceEthnicityCategory__: The racial or ethnic category of the individual.')
        st.write('- __AgeCategory__: The age category of the individual.')
        st.write('- __HeightInMeters__: The height of the individual in meters.')
        st.write('- __WeightInKilograms__: The weight of the individual in kilograms.')
        st.write('- __BMI__: Body Mass Index, a measure of body fat based on height and weight.')
        st.write('- __AlcoholDrinkers__: Information about the individual alcohol consumption.')
        st.write('- __HIVTesting__: Whether the individual has undergone HIV testing.')
        st.write('- __FluVaxLast12__: Whether the individual received a flu vaccine in the last 12 months.')
        st.write('- __PneumoVaxEver__: Whether the individual has ever received a pneumonia vaccine.')
        st.write('- __TetanusLast10Tdap__: Whether the individual had a tetanus vaccine in the last 10 years.')
        st.write('- __HighRiskLastYear__: Whether the individual has been at high risk for a health issue in the last year.')
        st.write('- __CovidPos__: Whether the individual has tested positive for COVID-19.')
        st.write('- __PersonCondition(BMI)__: A condition related to the individual BMI.')

        st.markdown(""" What subject does the dataset cover? According to the CDC, heart disease is a leading cause of death for people of most races in the U.S
        African Americans, American Indians and Alaska Natives, and whites). About half of all Americans (47%) have at least 1 of 3 major risk factors for heart disease: high blood pressure, high cholesterol,
        and smoking. Other key indicators include diabetes status, obesity (high BMI), not getting enough physical activity, or drinking too much alcohol. Identifying and preventing the factors that have 
        the greatest impact on heart disease is very important in healthcare. In turn, developments in computing allow the application of machine learning methods to detect "patterns" in the data that can predict a patient's condition""")
        st.markdown("[Link data](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)")
        st.markdown("--"*50)
        st.markdown("This is the sample of dataframe :")
        sample=st.dataframe(df.sample(25))
        btn=st.button("Display another sample")
        if btn:
            print(sample)
        st.markdown("-----------------------------------")
    #####################################################################################################################################
    num = df.describe()
    cat = df.describe(include="O")
    if add_selectbox =="Describtive Stats":
        st.subheader("Numerical Describtive Statistics")
        st.dataframe(num)
        st.markdown('*'*50)

        st.subheader("Numerical Describtive Statistics")
        st.dataframe(cat.T,width=800,height=700)
        st.markdown('*'*50)
    #####################################################################################################################################
    if add_selectbox =="EDA":
        st.subheader("In Exploratory data analysis (EDA) we have 3 type")
        st.markdown("1) Univariate")
        st.markdown("2) Bivariate")
        st.markdown("3) Multivariate")
        sb=st.selectbox("__Select what type to show visualization it__",["Univariate","Bivariate","Multivariate"])
        #########################   
        if sb== "Univariate":
            # fig=px.histogram(df , x='State',text_auto="0.2s",width=1500,height=600)
            # fig.update_traces(textfont_size=12,textposition="outside")
            # fig.update_layout(title_text="count people in State",title_x=0.5)
            # st.plotly_chart(fig)
            # st.markdown('- __note:__ the state with the most values is Washington')
            # st.markdown('*'*50)



            col1 = st.radio(
                "Select column you want to know about",
                ["State","AgeCategory", "Sex",],
                horizontal= True, )
            def hist_plot_uni(df, dim):
                fig_hist = px.histogram(df, x= dim,
                            width=1000, height=600,
                            text_auto="0.2s",
                            title= f'Distribution of {dim}')
                fig_hist.update_traces(textfont_size=12,textposition="outside")
                return fig_hist 
                
            st.plotly_chart(hist_plot_uni(df,col1))
            st.markdown('*'*50)
            

            # fig=px.histogram(df , x='AgeCategory',text_auto="0.2s",width=1000,height=600)
            # fig.update_traces(textfont_size=12,textposition="outside")
            # fig.update_layout(title_text="Age category",title_x=0.5)
            # st.plotly_chart(fig)
            # st.markdown('- __note:__ the most category is from 65 to 69')
            # st.markdown('*'*50)

            # col1,col2,col3=st.columns([5,1,5])
            # with col1:
            #     fig=px.histogram(df , x='Sex',text_auto="0.2s",width=500 , height=500)
            #     fig.update_traces(textfont_size=12,textposition="outside")
            #     fig.update_layout(title_text="gender",title_x=0.5)
            #     st.plotly_chart(fig)
            #     st.markdown('- __note:__ more female than male')
            #     st.markdown('*'*50)
            # with col3:
            #     fig=px.histogram(df , x='PhysicalActivities',text_auto="0.2s",width=500 , height=500)
            #     fig.update_traces(textfont_size=12,textposition="outside")
            #     fig.update_layout(title_text="people who do phiscal activities",title_x=0.5)
            #     st.plotly_chart(fig)
            #     st.markdown('- __note:__ most people do physical activities')
            #     st.markdown('*'*50)

            #pie chart
            col1,col2,col3=st.columns([1,5,1])
            with col2:
                col1 = st.radio(
                    "Select column you interest",
                    ["GeneralHealth", "LastCheckupTime", "RemovedTeeth","HadDiabetes"],
                    horizontal= True, )
                def pie_dist (data,dim):
                    dff = df[col1].value_counts()
                    fig_pie = px.pie(dff, values= dff.values, names= dff.index,hole=0.1, title= f'pie chart for {col1}')
                    return fig_pie
                st.plotly_chart(pie_dist (df,col1))
                st.markdown('*'*50)

            # col1,col2,col3=st.columns([5,2,5])
            # with col1:
            #     dff = df['GeneralHealth'].value_counts()
            #     fig=px.pie(data_frame=dff,
            #             names=dff.index,
            #             values=dff.values,
            #             hole=0.1)
            #     fig.update_layout(title_text="perecentage of general health",title_x=0.2)
            #     st.plotly_chart(fig)
            #     st.markdown('- __note:__ most value is very good')
            # with col3:
            #     dff = df['LastCheckupTime'].value_counts()
            #     fig=px.pie(data_frame=dff,
            #             names=dff.index,
            #             values=dff.values,
            #             hole=0.1)
            #     fig.update_layout(title_text="last cheack up time",title_x=0.2)
            #     st.plotly_chart(fig)
            #     st.markdown('- __note:__ most value is Within past year (anytime less than 12 months ago)')
            # st.markdown('*'*50)

            # col1,col2,col3=st.columns([5,2,5])
            # with col1:
            #     dff=df['RemovedTeeth'].value_counts()
            #     fig=px.pie(data_frame=dff,
            #                 names=dff.index,
            #                 values=dff.values,
            #                 hole=0.1)
            #     fig.update_layout(title_text="people who remove teeth",title_x=0.2)
            #     st.plotly_chart(fig)
            #     st.markdown('- __note:__ most value is people that remove teeth')
            #     st.markdown('*'*50)
            # with col3:
            #     dff=df['HadDiabetes'].value_counts()
            #     fig=px.pie(data_frame=dff,
            #                 names=dff.index,
            #                 values=dff.values,
            #                 hole=0.1)
            #     fig.update_layout(title_text="pie chart to shows percentage of values in HadDiabetes column",title_x=0.2)
            #     st.plotly_chart(fig)
            #     st.markdown('- __note:__ most value is people that __not have diabetes__')
            #     st.markdown('*'*50)



            fig = make_subplots(rows=3 , cols = 3 , subplot_titles = ('HadAngina',
                'HadStroke','HadAsthma','HadSkinCancer','HadCOPD',
                'HadDepressiveDisorder','HadKidneyDisease','HadArthritis','HadDiabetes',
                ))
            fig.add_trace(go.Histogram(x=df['HadAngina'],name='HadAngina'),row = 1 ,col = 1)
            fig.add_trace(go.Histogram(x=df['HadStroke'] ,name='HadStroke'),row = 1 ,col = 2)
            fig.add_trace(go.Histogram(x=df['HadAsthma'] ,name='HadAsthma'),row = 1 ,col = 3)
            fig.add_trace(go.Histogram(x=df['HadSkinCancer'] ,name='HadSkinCancer'),row = 2 ,col =1 )
            fig.add_trace(go.Histogram(x=df['HadCOPD'] ,name='HadCOPD'),row = 2 ,col = 2)
            fig.add_trace(go.Histogram(x=df['HadDepressiveDisorder'] ,name='HadDepressiveDisorder'),row = 2 ,col = 3)
            fig.add_trace(go.Histogram(x=df['HadKidneyDisease'] ,name='HadKidneyDisease'),row = 3 ,col = 1)
            fig.add_trace(go.Histogram(x=df['HadArthritis'],name='HadArthritis' ),row = 3 ,col = 2)
            fig.add_trace(go.Histogram(x=df['HadDiabetes'] ,name='HadDiabetes'),row = 3 ,col = 3)
            fig.update_layout(height=1000, width=1800, title_text="subplots of disease",title_x=0.4)
            st.plotly_chart(fig)
            st.markdown(' __note:__ most value is __NO__ in all graphs')
            st.markdown('*'*50)

            fig = make_subplots(rows=3 , cols = 2 , subplot_titles = ('DeafOrHardOfHearing', 'BlindOrVisionDifficulty',
        'DifficultyConcentrating', 'DifficultyWalking',
        'DifficultyDressingBathing', 'DifficultyErrands'))
            fig.add_trace(go.Histogram(x=df['DeafOrHardOfHearing'],name='DeafOrHardOfHearing' ),row = 1 ,col = 1)
            fig.add_trace(go.Histogram(x=df['BlindOrVisionDifficulty'] ,name='BlindOrVisionDifficulty'),row = 1 ,col = 2)
            fig.add_trace(go.Histogram(x=df['DifficultyConcentrating'] ,name='DifficultyConcentrating'),row = 2 ,col = 1)
            fig.add_trace(go.Histogram(x=df['DifficultyWalking'] ,name='DifficultyWalking'),row = 2 ,col =2 )
            fig.add_trace(go.Histogram(x=df['DifficultyDressingBathing'] ,name='DifficultyDressingBathing'),row = 3 ,col = 1)
            fig.add_trace(go.Histogram(x=df['DifficultyErrands'] ,name='DifficultyErrands'),row = 3 ,col = 2)
            fig.update_layout(height=1000, width=1500, title_text="subplots to the another diseases",title_x=0.4)
            st.plotly_chart(fig)
            st.markdown(' __note:__ most value is __NO___ in all __disease__')
            st.markdown('*'*50)

            fig = make_subplots(rows=2 , cols = 2 , subplot_titles = ('SmokerStatus' , 'ECigaretteUsage' , 'AlcoholDrinkers'))
            fig.add_trace(go.Histogram(x=df['SmokerStatus'] ,name='SmokerStatus'),row = 1 ,col = 1)
            fig.add_trace(go.Histogram(x=df['ECigaretteUsage'] ,name='ECigaretteUsage'),row = 1 ,col = 2)
            fig.add_trace(go.Histogram(x=df['AlcoholDrinkers'],name='AlcoholDrinkers' ),row = 2 ,col = 1)
            fig.update_layout(height=800, width=1500, title_text="subplots to the state of smoking , ECigarette and AlcoholDrinkers",title_x=0.4)
            st.plotly_chart(fig)
            st.markdown('__Note : NS is Never smoked , SD is smoking for some days , FS is Former smoker , ED is smoking for every days__')
            st.markdown('__Note : NUisNever used , SDisused some days , N/A (RN is Not at all (right now) ,ED is used every days__ ')
            st.markdown('in the first graph most value about smoker status is __NS__ never smoking ')
            st.markdown('in the second graprh most value about ECigarette usage is __NU__ never used ')
            st.markdown('in the third graprh most value about AlcoholDrinkers is __yes__')
            st.markdown('*'*50)


            # col1,col2,col3=st.columns([5,2,5])
            # with col1:
            #     fig=px.histogram(df , x='ChestScan',text_auto="0.2s",width=500 , height=500)
            #     fig.update_traces(textfont_size=12,textposition="outside")
            #     fig.update_layout(title_text="count people do chest scan or not",title_x=0.4)
            #     st.plotly_chart(fig)
            #     st.markdown('- __note:__ the most value is __NO__')
            #     st.markdown('*'*50)
            # with col3:
            #     fig=px.histogram(df , x='RemovedTeeth',text_auto="0.2s",width=500 , height=500)
            #     fig.update_traces(textfont_size=12,textposition="outside")
            #     fig.update_layout(title_text="histogram for people removed teeth",title_x=0.4)
            #     st.plotly_chart(fig)
            #     st.markdown('- __note:__ most value is __none of them__ that is peaople do not remove teeth')
            #     st.markdown('*'*50)

            col1,col2,col3=st.columns([5,2,5])
            with col1:
                fig = px.box(df, x ='BMI' )
                fig.update_layout(title_text='box plot for BMI',title_x=0.4)
                st.plotly_chart(fig)
                st.markdown("- __note:__ there is density from __24.13__ to __31.75__")

            with col3:
                fig=px.histogram(df , x='PersonCondition(BMI)',text_auto="0.2s",width=500 , height=500)
                fig.update_traces(textfont_size=12,textposition="outside")
                fig.update_layout(title_text="Count of People by BMI Condition",title_x=0.4)
                st.plotly_chart(fig)
                st.markdown('- UnderWeight is < __18.5__')
                st.markdown('- HealthyWeight is from __18.5__ to __24.99__')
                st.markdown('- OverWeight is from __25__ to __29.99__')
                st.markdown('- Obesity is >= __30__')
                st.markdown('- __note:__ most value is __Over weight__')
            st.markdown('*'*50)


            fig = make_subplots(rows=3 , cols = 2 , subplot_titles = ('HIVTesting', 'FluVaxLast12', 'PneumoVaxEver', 'TetanusLast10Tdap','HighRiskLastYear', 'CovidPos'))
            fig.add_trace(go.Histogram(x=df['HIVTesting'],name='HIVTesting' ),row = 1 ,col = 1)
            fig.add_trace(go.Histogram(x=df['FluVaxLast12'],name= 'FluVaxLast12'),row = 1 ,col = 2)
            fig.add_trace(go.Histogram(x=df['PneumoVaxEver'] ,name='PneumoVaxEver'),row = 2 ,col = 1)
            fig.add_trace(go.Histogram(x=df['TetanusLast10Tdap'],name='TetanusLast10Tdap' ),row = 2 ,col = 2)
            fig.add_trace(go.Histogram(x=df['HighRiskLastYear'] ,name='HighRiskLastYear'),row = 3 ,col = 1)
            fig.add_trace(go.Histogram(x=df['CovidPos'] ,name='CovidPos'),row = 3 ,col = 2)
            fig.update_layout(height=1000, width=1500, title_text="subplots show if you take the vaccine or not",title_x=0.5)
            st.plotly_chart(fig)
            st.markdown("HIVTesting graph the most value is __NO__")
            st.markdown("FluVaxLast12 graph the most value is __YES__")
            st.markdown("PneumoVaxEver graph the most value is __NO__")
            st.markdown("TetanusLast10Tdap graph the most value is __NO-Tet__  it is not take vaccine")
            st.markdown("HighRiskLastYear graph the most value is __NO__") 
            st.markdown("CovidPos graph the most value is __NO__")
            st.markdown('*'*50)


            col1,col2,col3=st.columns([1,6,1])
            with col2:
                fig=px.histogram(df , x='HadHeartAttack',text_auto="0.2s",width=600 , height=600)
                fig.update_traces(textfont_size=12,textposition="outside")
                fig.update_layout(title_text="Count of People have heart attack disease or not",title_x=0.3)
                st.plotly_chart(fig)
                st.markdown('- __note:__ most value __NO__ that is people __not have heart attack disease__ ')
                st.markdown('*'*50)
    ######################################################################################################################

        if sb== "Bivariate":
            st.header("1-Heatmap to show correlation between the numeic columns")
            df_numeric = df.select_dtypes(exclude='O')
            fig=sns.heatmap(df_numeric.corr(),annot=True)
            fig.set_title("Heatmap for numeric columns ")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            st.markdown("__Note:__ there is posetive correlation between __WeightInKilograms__ and __BMI__")
            st.markdown("*"*50)


            st.header(" Q-1 what is the correlation between WeightInKilograms and BMI ?")
            fig = px.scatter(df , x='WeightInKilograms',y='BMI')
            fig.update_layout(title_text="Correlation between WeightInKilograms and BMI")
            st.plotly_chart(fig)
            st.markdown("- __Note:__ there is __positive__ correlation between __WeightInKilograms__ and __BMI__")
            st.markdown("*"*50)


            st.header("Q-2 What is the top 10 state have  Heart Attack ?")
            #pandas
            df0=df[df["HadHeartAttack"]=="Yes"]
            df1=df0.groupby('State')[['HadHeartAttack']].count().sort_values(by='HadHeartAttack',ascending=False).reset_index()
            fig=px.histogram(df1.head(10)  , x ='State',
                width=1200, height=600,
                y='HadHeartAttack' ,
                title='Distribution of higher Heart Attack by  to 10 State',
                text_auto=True
                )
            st.plotly_chart(fig)
            st.markdown("- __Note:__ most state has people that had heart attack disease is Washington")
            st.markdown("*"*50)


            st.header("Q-3 What are the states have lower Heart Attack ?")
            #pandas
            fig=px.histogram(df1.tail(10)  , x ='State',
                y='HadHeartAttack' ,
                width=1200, height=600,
                title='Distribution of low Heart Attack by  to 10 State',
                text_auto=True
                )
            st.plotly_chart(fig)
            st.markdown("- __Note:__ low state has people that had heart attack disease is Virgin Islands")
            st.markdown("*"*50)


            st.header("Q-4 How does the occurrence of heart attacks vary based on gender?")
            fig=px.histogram(df  , x ='Sex',
                color='HadHeartAttack' ,
                barmode ="group",
                width=1200, height=600,
                title='Distribution of Heart Attacks by Gender',
                text_auto=True
                )
            st.plotly_chart(fig)
            st.markdown("- __Note:__Number of __male__ had heart attack __bigger__ than __female__  ")
            st.markdown("*"*50)


            st.header(" Q-5 What is the distribution of 'GeneralHealth' ratings among individuals who had a heart attack compared to those who did not?")
            fig=px.histogram(df,x='GeneralHealth', 
                    color='HadHeartAttack', 
                    barmode='group',
                    barnorm="percent",
                    width=1200, height=600,
                    text_auto=True ,
                    title = 'Distribution of GeneralHealth Ratings Among Individuals with and without Heart Attacks',
                category_orders ={'GeneralHealth':['Excellent', 'Good', 'Very good', 'Poor', 'Fair']})
            st.plotly_chart(fig)
            st.markdown("- __Note:__ Most people have heart attacks , who are __poor__ ")
            st.markdown("*"*50)

            col1,col2=st.columns(2) 
            with col1:
                st.header("Q-6 Is there a correlation between the number of 'PhysicalHealthDays' and the likelihood of having a heart attack?")
                fig=px.box(df , x='HadHeartAttack' , 
                y='PhysicalHealthDays' ,
                title='PhysicalHealthDays vs. Likelihood of Having a Heart Attack',
                )
                st.plotly_chart(fig)
                st.markdown("- __Note:__ people who have physical health problems for last 30 days the ones who have the most heart attacks ")
                st.markdown("*"*50)

            with col2 :
                st.header("Q-7 Is there a correlation between the number of 'MentalHealthDays' and the likelihood of having a heart attack?")
                fig=px.box(df , x='HadHeartAttack' , 
                y='MentalHealthDays' ,
                title='MentalHealthDays vs. Likelihood of Having a Heart Attack',
                )
                st.plotly_chart(fig)
                st.markdown("- __Note:__ there is no connection between MentalHealthDays and Heart Attack")
                st.markdown("*"*50)




            st.header('Q-8 Do individuals who engage in PhysicalActivities regularly have a lower incidence of heart attacks?')
            fig=px.histogram(df,x='PhysicalActivities', 
                 color='HadHeartAttack', 
                 barmode='group',
                 barnorm="percent",
                 text_auto=True ,
                 title = 'Incidence of Heart Attacks Based on Physical Activities',
                )
            st.plotly_chart(fig)
            st.markdown("- __Note:__ most people at risk of heart attack do not Physical Activities ")
            st.markdown("*"*50)


            st.header("Q-9 How does RemovedTeeth relate to the occurrence of heart attacks?")
            fig=px.histogram(df,x='RemovedTeeth', 
                 color='HadHeartAttack', 
                 barmode='group',
                 barnorm="percent",
                 text_auto=True ,
                 title = 'Incidence of Heart Attacks Based on Removed Teeth',
                )
            st.plotly_chart(fig)
            st.markdown('- __Note:__ most people at risk of heart attack who remove teeth')
            st.markdown("*"*50)



            st.header("Q-10 Are Angine ,strok,asthma ,skin cancer , COPD ,DepressiveDisorder,KidneyDisease,Arthritis and Diabetes realated to people have a heart attack ?")
            fig = make_subplots(rows=3 , cols = 3 , subplot_titles = ('HadAngina',
            'HadStroke','HadAsthma','HadSkinCancer','HadCOPD',
            'HadDepressiveDisorder','HadKidneyDisease','HadArthritis','HadDiabetes',
            ))
            fig.add_trace(go.Histogram(x=df0['HadAngina'],name='HadAngina',bingroup='HadHeartAttack' ),row = 1 ,col = 1)
            fig.add_trace(go.Histogram(x=df0['HadStroke'] ,name='HadStroke'),row = 1 ,col = 2)
            fig.add_trace(go.Histogram(x=df0['HadAsthma'] ,name='HadAsthma'),row = 1 ,col = 3)
            fig.add_trace(go.Histogram(x=df0['HadSkinCancer'] ,name='HadSkinCancer'),row = 2 ,col =1 )
            fig.add_trace(go.Histogram(x=df0['HadCOPD'] ,name='HadCOPD'),row = 2 ,col = 2)
            fig.add_trace(go.Histogram(x=df0['HadDepressiveDisorder'] ,name='HadDepressiveDisorder'),row = 2 ,col = 3)
            fig.add_trace(go.Histogram(x=df0['HadKidneyDisease'] ,name='HadKidneyDisease'),row = 3 ,col = 1)
            fig.add_trace(go.Histogram(x=df0['HadArthritis'],name='HadArthritis' ),row = 3 ,col = 2)
            fig.add_trace(go.Histogram(x=df0['HadDiabetes'] ,name='HadDiabetes'),row = 3 ,col = 3)
            fig.update_layout(height=1000, width=1800, title_text="subplots of disease",title_x=0.4)
            st.plotly_chart(fig)
            st.markdown(' __Note:__ there is no correlation between these diseases and heart attack disease')
            st.markdown('*'*50)

            st.header("Q-11 Are individuals who smoke or ECigaretteUsage or AlcoholDrinkers more likely to have had a heart attack compared to non-smokers or ECigaretteUsage or AlcoholDrinkers?")
            col1 = st.radio(
                "Select column you interest",
                ["SmokerStatus", "ECigaretteUsage", "AlcoholDrinkers"],
                horizontal= True, )
            def hist_plot(df, dim):
                fig_hist = px.histogram(df, x= dim,
                            color='HadHeartAttack',
                            barmode='group',
                            barnorm="percent",
                            width=1000, height=600,
                            text_auto=True ,
                            title= f'Distribution of {dim} According to HadHeartAttack')
                fig_hist.update_layout(xaxis_title=col1,
                    yaxis_title="Percentage ")
                return fig_hist 
            st.plotly_chart(hist_plot(df,col1))
            st.markdown('*'*50)


            st.header('Q-12 Is there a relationship between age categories and the likelihood of having had a heart attack?')
            col1 = st.radio(
                "Select graph you interest",
                ["Boxplot", "Histograme"],
                horizontal= True, )
            if col1 == 'Boxplot' :
                fig = px.box(df , x='HadHeartAttack' ,
                y='AgeCategory',
                category_orders={'AgeCategory':['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older']},
                title="Association between Age Categories and Incidence of Heart Attacks"
                )
            else :
                fig=px.histogram(df,x='AgeCategory', 
                 color='HadHeartAttack', 
                 barmode='group',
                 barnorm="percent",
                 width=1500, height=600,
                 text_auto=True ,
                 title = 'Association between Age Categories and Incidence of Heart Attacks', 
                 )
                fig.update_layout(xaxis_title="AgeCategory",
                    yaxis_title="Percentage for people who has heart attack",
                 )
            st.plotly_chart(fig)
            st.markdown('*'*50)


            st.header("Q-13 Does BMI show a correlation with the occurrence of heart attacks?")
            dff= df0.groupby('PersonCondition(BMI)')[['HadHeartAttack']].count().sort_values(by='HadHeartAttack' ,ascending=False).reset_index()
            fig = px.histogram(dff , x='PersonCondition(BMI)',
             y='HadHeartAttack',
             text_auto=True ,
             width=1000, height=600,
             title="The Relationship Between BMI and the Risk of Heart Attacks",
             )
            st.plotly_chart(fig)
            st.markdown('*'*50)


            st.header("Q-14 Does HIV testing have any correlation with the occurrence of heart attacks?")
            dff= df0.groupby('HIVTesting')[['HadHeartAttack']].count().sort_values(by='HadHeartAttack' ,ascending=False).reset_index()
            fig = px.histogram(dff , x='HIVTesting' ,
             y = 'HadHeartAttack',
             title="Analyzing the Correlation Between HIV Testing History and Heart Attacks Among Affected Individuals"
            )
            st.plotly_chart(fig)
            st.markdown('*'*50)
#######################################################################################################################################
        # if sb== 'Multivariate':
        #     col_id =st.selectbox("__Select what type to show visualization it__",['State', 'Sex', 'GeneralHealth', 'PhysicalHealthDays',
        #                         'MentalHealthDays', 'LastCheckupTime', 'PhysicalActivities'])
                                
        #     # col_count =st.selectbox('__choose the second column__' , ['State', 'Sex', 'GeneralHealth', 'PhysicalHealthDays',
        #     #                     'MentalHealthDays', 'LastCheckupTime', 'PhysicalActivities',
        #     #                     ])
        #     def sunburst_chart (id,count):
        #         fig = go.Figure(go.Sunburst(
        #                     labels=id,
        #                     parents=count,
        #                     values='HadHeartAttack'
        #                     ))
        #     st.plotly_chart(sunburst_chart(col_id,count='LastCheckupTime'))

#######################################################################################################################################

    if add_selectbox=='Conclusion':
        st.header("Conclusion from analysis for this data")
        st.write('1- Our analysis indicates that higher PhysicalHealthDays and lower GeneralHealth are associated with an increased likelihood of heart attacks.')
        st.write('2-Regularity in PhysicalActivities appears to correlate with a reduced risk of heart attacks.')
        st.write('3-RemovedTeeth and  Arthritis disease are identified as potential risk factors for heart attacks.')
        st.write('4-Smoking appears as a major contributor to heart attacks')
        st.write('5-Age categories play a critical role, with the risk of heart attacks increasing with age.')
        st.write('6-Higher BMI values are associated with an getting likelihood of experiencing a heart attack.')
        st.write('7-there is weak correlation may indicate that HIV testing is not a strong predictor of heart attacks in our dataset.')
        st.markdown("*"*50)
        st.header("Finally:")
        st.write("""These results highlight that making healthy choices in our daily lives can help lower the chances of having a heart attack.
                  Things like staying active, quitting smoking,
                  and keeping a healthy weight play a critical role in reducing the risk of heart attacks. 
                 Taking care of our lifestyle can make a big difference in keeping our hearts healthy.""")










if __name__ == "__main__":
    cProfile.run("main()") 