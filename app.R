#mobile_price <- read.csv("/home/tcs/Desktop/1411148/r/shiny/final/train.csv", stringsAsFactors = FALSE)
require(shiny)
require(shinyIncubator)
library(shinydashboard)
library(googleVis)
require(pastecs)
require(caret)
require(e1071)
require(randomForest)
require(nnet)
require(glmnet)
require(gbm)
library(mice)
library(data.table)
library(VIM)
require(fastICA)
library("PASWR")
library(RColorBrewer)
require("doMC")
library(rpart)
library(rpart.plot)
library(corrplot)
library(ggplot2)
library(shinythemes)
library(party)
setwd(".")
#to install the package from the cran
#install.packages('googleVis', dependencies=TRUE, repos='http://cran.rstudio.com/')
ui <- fluidPage(shinyUI(dashboardPage(
  skin = "blue",
  dashboardHeader(title = "Predictive Analytics "),
  dashboardSidebar(
    sidebarMenu(
      
      menuItem("Summary", tabName = "summary", icon = icon("dashboard")),
      menuItem("Data Preparation", tabName = "datapreparation", icon = icon("wrench")),
      menuItem("Analysis", tabName = "analysis", icon = icon("cogs"),
               menuSubItem("Algorithm",icon = icon("cog"), tabName = "algorithm")
      ),
      menuItem("Results", tabName = "results", icon = icon("dashboard")),
      menuItem("User Interface",tabName ="ui" ,icon = icon("list-alt") ),
      
      tags$img(src='sk.png',height=300,width=240),
      menuItem(h4("Developed By")),
      menuItem(h4("ANJANA NARAYANA.K")),
      menuItem(h4("Roll:no-1221616103")),
      menuItem(h4("Branch:- DATA SCIENCE")),
      menuItem(h4("GITAM UNIVERSITY"))
              
      
    )
  ),
  dashboardBody(
    tabItems(
      tabItem(tabName = "summary", 
              fluidRow(
                box(title = "How To Use", status = "info", solidHeader = TRUE,
                    collapsible = TRUE, width = 8,
                   
                    h4("Step 1: Upload Dataset"),
                    h5("Ideally any csv file is useable.  It is recommended to perform cleaning and munging methods prior to the upload though. We intend to apply data munging/cleaning methods in this app in the near future."),
                    h4("Step 2: Analyze Data"),
                    h5("Current version allows the user to perform basic missing analysis."),
                    h4("Step 3: Choose Pre-processing Methods"),
                    h5("Basic K-Cross Validation Methods are applicable. "),
                    h4("Step 4: Choose Model"),
                    h5("Choose from a selection of machine learning models to run.  Selected parameters for each corresponding model are available to tune and manipulate."),
                    h4("Step 5: Run Application"),
                    h5("Once the model(s) have been executed, the results for each model can be viewed in the results tab for analysis.")
                )),
              
              fluidRow(
                box(title = "Libraries/Dependencies",status = "info", solidHeader = TRUE,
                    collapsible = TRUE, width = 8,
                   
                    h4("- The caret package was used for the backend machine learning algorithms."),
                    h4("- Shiny Dashboard was used for the front end development."),
                    h4("- The application is compatiable with AWS for server usage.")
                     )
              ),
              
                fluidRow(
                box(title = "This application is for.....",status = "info", solidHeader = TRUE,
                    collapsible = TRUE, width = 8,
                    h4("- O/P: we need to train the data to predict the price range for the new features of mobile."),
                    h4("- We have data of mobile price ranges along with  features in it."),
                    h4("- Using  a Decision Tree Classification Model")
                )
              )
             
      ),
      tabItem(tabName = "datapreparation",
              fluidPage(
                tabBox(
                  id = "datapreptab",
                  
                  tabPanel(h4("Summary of the Data set"), verbatimTextOutput("summary")),
                 
                  tabPanel(h4("first 5 rows of the dataframe"),tableOutput("head")),
                  
                  tabPanel(h4("Data"),
                           fileInput("file", "Choose CSV File",
                                     multiple = TRUE,
                                     accept = c("text/csv",
                                                "text/comma-separated-values,text/plain",
                                                ".csv")),
                           
                           tags$hr(),
                           checkboxInput("header", "Header", TRUE),
                           radioButtons("sep", "Separator",
                                        choices = c(Comma = ",",
                                                    Semicolon = ";",
                                                    Tab = "\t"),
                                        selected = ","),
                           
                           # Input: Select quotes ----
                           radioButtons("quote", "Quote",
                                        choices = c(None = "",
                                                    "Double Quote" = '"',
                                                    "Single Quote" = "'"),
                                        selected = '"'),
                           tags$hr(),
                           radioButtons("disp", "Display",
                                        choices = c(Head = "head",
                                                    All = "all"),
                                        selected = "head")
                           )
                           
                )
              )
      ),
      tabItem(tabName = "algorithm",
              fluidRow(
                box(title = "Decision Tree Algorithm", status = "primary", solidHeader = TRUE, collapsible = TRUE, width = 11,
                    
                    
                    h4("Decision tree learning uses a decision tree (as a predictive model) to go from observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves). It is one of the predictive modelling approaches used in statistics, data mining and machine learning. Tree models where the target variable can take a discrete set of values are called classification trees; in these tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees.."),
                    uiOutput("DSTmodelParametersUI"),
                    
                    tags$hr()
                )
              ),
              
              actionButton('rpart',label = 'Decision Tree',icon("leaf",lib = "glyphicon"),
                           style="color: #fff; background-color: green;border-color: #2e6da4")),
      tabItem(tabName = "results",
              fluidPage(
                title = "Decision Tree", status = "primary", solidHeader = TRUE, collapsible = TRUE, width = 11,
                    
                    tabsetPanel(
                      tabPanel("Best Results",tableOutput("result"),icon = icon("table")),
                      tabPanel("confusion matrix",tags$img(src="accuracy.png")),
                      tabPanel("Decision tree plot",plotOutput("plot"),icon = icon("bar-chart-o")),
                      #tabPanel("User I/P Results",tableOutput("us_result"),icon = icon("table")),
                      
                      tabPanel("Summary of the Model",verbatimTextOutput("summary1"),icon = icon("list-alt")),
                      
                      tabPanel("Correlation Plot",plotOutput("correlation"),icon = icon("bar-chart-o")))
                )),
      tabItem(tabName = "ui",
              fluidPage(
                
                fluidRow(
                  titlePanel(title ="Predictive Analytics on Mobile Price Range"),
                  sidebarLayout(
                    sidebarPanel(
                      actionButton('run_model',label = 'RUN MODEL',icon("leaf",lib = "glyphicon"),
                                   style="color: #fff; background-color:green;border-color: #2e6da4"),
                      
                  
                      tags$style(HTML(".js-irs-5 .irs-single, .js-irs-5 .irs-bar-edge, .js-irs-5 .irs-bar {background: purple}")),
                      tags$style(HTML(".js-irs-3 .irs-single, .js-irs-3 .irs-bar-edge, .js-irs-3 .irs-bar {background: orange}")),
                      tags$style(HTML(".js-irs-6 .irs-single, .js-irs-6 .irs-bar-edge, .js-irs-6 .irs-bar {background: pink}")),
                      tags$style(HTML(".js-irs-8 .irs-single, .js-irs-8 .irs-bar-edge, .js-irs-8 .irs-bar {background: green}")),
                      # tags$style(HTML(".js-irs- .irs-single, .js-irs-5 .irs-bar-edge, .js-irs-5 .irs-bar {background: blue}")),
                      
                      sliderInput("battery_powerInput","Battery Power",min=500,max=2000,value=1400,step =1 ,pre = "mAh"),
                      selectInput("blueInput", "Blue", choices = c("0", "1"),selected = "0"),
                      sliderInput("clock_speedInput","Clock Speed",min=0,max=4,value=0.5,step=0.1,pre = "ghz"),
                      selectInput("dual_simInput", "Dual Sim", choices = c("0", "1"),selected = "0"),
                      sliderInput("fcInput","Fast charging",0,20,10,pre = "min"),
                      selectInput("four_gInput", "Four_g", choices = c("0", "1"),selected = "0"),
                      sliderInput("int_memoryInput","int_memory",0,70,30,step=1,pre = "gb"),
                      sliderInput("m_depInput","m_depth",0,1,0.5,step=0.1),
                      sliderInput("mobile_wtInput","Mobile Width",80,200,150,step = 1,pre = "g"), 
                      sliderInput("n_coresInput","Cores",1,8,5),
                      sliderInput("pc","Screen Height",0,20,15),
                      sliderInput("px_heightInput","pixel Height",0,1960,300,step = 1), 
                      sliderInput("px_widthInput","Pixel Width",498,2000,1500,step=1),
                      sliderInput("ramInput","Ram",256,4000,3000,step=1,pre = "mb"),
                      sliderInput("sc_hInput","Screen Height",0,20,15,step = 1,pre = "cm"),
                      sliderInput("sc_wInput","Screen Width",0,20,15,step=1,pre = "cm"),
                      sliderInput("talk_timeInput","Talk Time",0,20,15,step=1,pre = "hr"),
                      selectInput("three_gInput", "Three(G)", choices = c("0", "1"),selected = "0"),
                      selectInput("touch_screenInput", "Touch Screen", choices = c("0", "1"),selected = "0"),
                      selectInput("wifiInput", "wifi", choices = c("0", "1"),selected = "0")
                      
                    ),
                    
                    
                    mainPanel("User Given values",
                              
                              tabsetPanel(
                                tabPanel("Test_Data ",tableOutput("test")),
                               
                                 
                                tabPanel("Mobile_price (Target class)",tableOutput("results")),
                                tabPanel("Class Ranges",tags$img(src="df.png"))
                              
                              ) )
                  )
                )
              )
      )
    )
    
    
  ))))


server <- function(input, output) {
 
mobile_price <- read.csv("train.csv", stringsAsFactors = FALSE)
  
output$mobile_price<-renderTable({
    req(input$file)
    df <- read.csv(input$file$datapath,
                   header = input$header,
                   sep = input$sep,
                   quote = input$quote)
    if(input$dips=="head"){
      return(as.data.frame(df))
    }
  })
    output$head <- renderTable({
    
    req(input$file)
    
    df <- read.csv(input$file$datapath,
                   header = input$header,
                   sep = input$sep,
                   quote = input$quote)
    
    if(input$disp == "head") {
      return(head(df))
    }
    else {
      return(df)
    }
    
  })
   output$summary <- renderPrint(
     {
       req(input$file)
       
       df <- read.csv(input$file$datapath,
                      header = input$header,
                      sep = input$sep,
                      quote = input$quote)
       
       if(input$disp == "head") {
         return(summary(df))
       }
       else {
         return(df)
       }
     }
   )
  
  set.seed(1000)    
  observe({
  
    intrain=createDataPartition(y=mobile_price$price_range,p=0.7,list=FALSE)
    
    training=mobile_price[intrain,]
    testing=mobile_price[-intrain,]
    #testing=user_data[1,]
    observeEvent(input$rpart,{
      ml_rpart<-rpart(training$price_range~.,method='class',data=training,control=rpart.control(minsplit=8,cp=0))
      model_pred<-predict(ml_rpart, testing, type="class")
      
      
      output$correlation<-renderPlot({
        df <- read.csv("trainc.csv", stringsAsFactors = FALSE)
        corrplot(cor(df))
      })
      output$result<-renderTable({
        table(model_pred, testing$price_range)    })
      output$cm<-renderTable({
        table()
      })
      output$summary1 <- renderPrint(summary(ml_rpart))
      output$battery_powerInput <- renderPrint({ input$battery_powerInput })
      #output$summary <- renderPrint(head(summary(mobile_price)))
      
      
      output$plot <- renderPlot({
        prp(ml_rpart,box.palette = "green",tweak = 0.99,cex=.7, main="Pruned Classification Tree for Mobile Price  data")
        # prune the treefirst then plot the pruned tree 
        
        
      })
    })
  })
  observe({
    battery_power =as.integer(input$battery_powerInput)
    blue= as.integer(input$blueInput)
    clock_speed=as.numeric(input$clock_speedInput)
    dual_sim= as.integer(input$dual_simInput)
    fc = as.integer(input$fcInput)           
    four_g=as.integer(input$four_gInput)         
    int_memory=as.integer(input$int_memoryInput)
    m_dep=  as.numeric(input$m_depInput)
    mobile_wt= as.integer(input$mobile_wtInput)
    n_cores=as.integer(input$n_coresInput)
    pc=as.integer(input$pc)
    px_height=as.integer(input$px_heightInput)
    px_width= as.integer(input$px_widthInput)
    ram=as.integer(input$ramInput)
    sc_h=as.integer(input$sc_hInput)
    sc_w=as.integer(input$sc_wInput)
    talk_time=as.integer(input$talk_timeInput)
    three_g= as.integer(input$three_gInput) 
    touch_screen= as.integer(input$touch_screenInput)
    wifi= as.integer(input$wifiInput)
    
    test<-cbind( battery_power, blue,clock_speed,dual_sim, fc,four_g,int_memory,m_dep,mobile_wt, n_cores,pc,
                 px_height,px_width, ram,sc_h,sc_w,talk_time,three_g,touch_screen,wifi )
    test<-as.data.frame(test)
    test$price_range<-""
    observeEvent(input$run_model,{
      
      run_model<-rpart(training$price_range~.,method='class',data=training,control=rpart.control(minsplit=8,cp=0))
      
      
      prediction<-predict(run_model, newdata=test, type="class")
      
      output$results<-renderTable(prediction)
      
      output$test<-renderTable(test)
      
    })
  })
  
  
}

shinyApp(ui = ui, server = server)

