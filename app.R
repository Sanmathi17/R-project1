library(shiny)
library(tidymodels)
library(tidyverse)
library(ranger)
library(kernlab)
library(DT)
library(plotly)

# UI Definition
ui <- fluidPage(
  titlePanel("Machine Learning Model Comparison"),
  
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "Upload CSV File",
                accept = c("text/csv", "text/comma-separated-values,text/plain", ".csv")),
      
      selectInput("target", "Select Target Variable", choices = NULL),
      
      numericInput("train_ratio", "Training Data Ratio", 
                   value = 0.8, min = 0.5, max = 0.9, step = 0.1),
      
      checkboxGroupInput("models", "Select Models to Compare:",
                         choices = c("Random Forest" = "rf",
                                     "SVM" = "svm",
                                     "Logistic Regression" = "logistic"),
                         selected = c("rf", "svm", "logistic")),
      
      actionButton("analyze", "Analyze Data", class = "btn-primary")
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Data Preview",
                 h4("Dataset Preview"),
                 DTOutput("data_preview"),
                 h4("Data Summary"),
                 verbatimTextOutput("data_summary")),
        
        tabPanel("Model Comparison",
                 h4("Model Performance Metrics"),
                 plotlyOutput("performance_plot"),
                 DTOutput("metrics_table")),
        
        tabPanel("Feature Importance",
                 h4("Variable Importance (Random Forest)"),
                 plotlyOutput("importance_plot")),
        
        tabPanel("Predictions",
                 h4("Model Predictions on Test Set"),
                 DTOutput("predictions_table"))
      )
    )
  )
)

# Server Logic
server <- function(input, output, session) {
  # Reactive values to store data and results
  rv <- reactiveValues(
    data = NULL,
    train_data = NULL,
    test_data = NULL,
    results = NULL,
    predictions = NULL
  )
  
  # Update target variable choices when data is uploaded
  observeEvent(input$file, {
    req(input$file)
    rv$data <- read.csv(input$file$datapath)
    updateSelectInput(session, "target",
                      choices = names(rv$data))
  })
  
  # Data preprocessing function
  preprocess_data <- function(data, target_col, train_ratio) {
    # Convert target to factor for classification
    data[[target_col]] <- as.factor(data[[target_col]])
    
    # Create train/test split
    set.seed(123)
    split <- initial_split(data, prop = train_ratio, strata = target_col)
    train_data <- training(split)
    test_data <- testing(split)
    
    # Create recipe
    recipe <- recipe(formula(paste(target_col, "~ .")), data = train_data) %>%
      step_normalize(all_numeric_predictors()) %>%
      prep()
    
    # Process data
    train_processed <- bake(recipe, new_data = train_data)
    test_processed <- bake(recipe, new_data = test_data)
    
    return(list(train = train_processed, test = test_processed))
  }
  
  # Model training and evaluation
  train_and_evaluate <- function(train_data, test_data, target_col, models) {
    results <- list()
    predictions <- data.frame(Actual = test_data[[target_col]])
    
    if ("rf" %in% models) {
      rf_model <- ranger(
        formula = formula(paste(target_col, "~ .")),
        data = train_data,
        importance = "impurity"
      )
      rf_pred <- predict(rf_model, data = test_data)
      results$rf <- list(
        accuracy = mean(rf_pred$predictions == test_data[[target_col]]),
        importance = rf_model$variable.importance
      )
      predictions$RF_Pred <- rf_pred$predictions
    }
    
    if ("svm" %in% models) {
      svm_model <- ksvm(
        formula(paste(target_col, "~ .")),
        data = train_data,
        kernel = "rbfdot"
      )
      svm_pred <- predict(svm_model, newdata = test_data)
      results$svm <- list(
        accuracy = mean(svm_pred == test_data[[target_col]])
      )
      predictions$SVM_Pred <- svm_pred
    }
    
    if ("logistic" %in% models) {
      log_model <- glm(
        formula(paste(target_col, "~ .")),
        data = train_data,
        family = "binomial"
      )
      log_pred <- predict(log_model, newdata = test_data, type = "response")
      log_pred_class <- ifelse(log_pred > 0.5, 
                               levels(train_data[[target_col]])[2],
                               levels(train_data[[target_col]])[1])
      results$logistic <- list(
        accuracy = mean(log_pred_class == test_data[[target_col]])
      )
      predictions$Logistic_Pred <- log_pred_class
    }
    
    return(list(results = results, predictions = predictions))
  }
  
  # Analyze button action
  observeEvent(input$analyze, {
    req(rv$data, input$target)
    
    # Preprocess data
    processed_data <- preprocess_data(rv$data, input$target, input$train_ratio)
    rv$train_data <- processed_data$train
    rv$test_data <- processed_data$test
    
    # Train and evaluate models
    model_results <- train_and_evaluate(rv$train_data, rv$test_data, 
                                        input$target, input$models)
    rv$results <- model_results$results
    rv$predictions <- model_results$predictions
  })
  
  # Render data preview
  output$data_preview <- renderDT({
    req(rv$data)
    datatable(head(rv$data, 100), options = list(scrollX = TRUE))
  })
  
  # Render data summary
  output$data_summary <- renderPrint({
    req(rv$data)
    summary(rv$data)
  })
  
  # Render performance plot
  output$performance_plot <- renderPlotly({
    req(rv$results)
    accuracies <- sapply(rv$results, function(x) x$accuracy)
    plot_data <- data.frame(
      Model = names(accuracies),
      Accuracy = accuracies
    )
    
    plot_ly(plot_data, x = ~Model, y = ~Accuracy, type = "bar",
            text = ~sprintf("%.3f", Accuracy),
            textposition = "auto") %>%
      layout(title = "Model Accuracy Comparison",
             yaxis = list(range = c(0, 1)))
  })
  
  # Render metrics table
  output$metrics_table <- renderDT({
    req(rv$results)
    accuracies <- sapply(rv$results, function(x) x$accuracy)
    data.frame(
      Model = names(accuracies),
      Accuracy = round(accuracies, 4)
    )
  })
  
  # Render importance plot
  output$importance_plot <- renderPlotly({
    req(rv$results$rf)
    importance_data <- data.frame(
      Feature = names(rv$results$rf$importance),
      Importance = rv$results$rf$importance
    ) %>%
      arrange(desc(Importance))
    
    plot_ly(importance_data, x = ~reorder(Feature, Importance), 
            y = ~Importance, type = "bar") %>%
      layout(title = "Feature Importance",
             xaxis = list(title = "Feature"),
             yaxis = list(title = "Importance"))
  })
  
  # Render predictions table
  output$predictions_table <- renderDT({
    req(rv$predictions)
    datatable(rv$predictions, options = list(scrollX = TRUE))
  })
}

# Run the app
shinyApp(ui = ui, server = server)