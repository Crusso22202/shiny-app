#Libraries----

library(data.table)
library(xgboost)
library(caret)
library(shiny)
library(data.table)
library(xgboost)
install.packages("leaflet")
install.packages("httr")
install.packages("jsonlite")
library(leaflet)
library(httr)
library(jsonlite)
install.packages("shinydashboard")
library(shinydashboard)
install.packages("tidygeocoder")
library(tidygeocoder)

#School District----

# Define paths to zip files and extraction directories
zip_files <- list(
  elementary = "path/to/tl_2024_06_elsd.zip",
  secondary = "path/to/tl_2024_06_scsd.zip",
  unified = "path/to/tl_2024_06_unsd.zip"
)

extraction_paths <- list(
  elementary = "path/to/elementary/",
  secondary = "path/to/secondary/",
  unified = "path/to/unified/"
)

# Unzip all files
lapply(names(zip_files), function(key) {
  unzip(zip_files[[key]], exdir = extraction_paths[[key]])
})

# Load the shapefiles into GeoDataFrames
elementary_districts <- st_read(file.path(extraction_paths$elementary, "tl_2024_06_elsd.shp"))
secondary_districts <- st_read(file.path(extraction_paths$secondary, "tl_2024_06_scsd.shp"))
unified_districts <- st_read(file.path(extraction_paths$unified, "tl_2024_06_unsd.shp"))

# Ensure all GeoDataFrames have the same CRS
elementary_districts <- st_transform(elementary_districts, crs = 4326)
secondary_districts <- st_transform(secondary_districts, crs = 4326)
unified_districts <- st_transform(unified_districts, crs = 4326)

# Combine all district data
all_districts <- bind_rows(elementary_districts, secondary_districts, unified_districts)

# Print the total number of districts
cat("Total Number of School Districts:", nrow(all_districts), "\n")
#Model----

# Load the dataset
df <- lotwizeV6
crime_data <- crime_index

# Convert zipcode_median_income to data.table
setDT(zipcode_median_income)

# Convert to data.table
setDT(df)

# Rename the column
setnames(df, old = "Crime.Index_y", new = "Crime Index")
setnames(df, old = "School.District.Rating_y", new = "School District Rating")

df_model <- df[, .(price, `address.city`, bedrooms, bathrooms, homeType, lotSize, `resoFacts.yearBuilt`,
                   `LivingArea...LotSize`, longitude, latitude, school_district, `address.streetAddress`,
                   `address.zipcode`, distanceHospital, distanceSchool, distanceMall, beachProximity,
                   median_income, elevation, relative_elevation, `Crime Index`, `School District Rating`)]

# Remove NA values
df_model <- na.omit(df_model)

# Define categorical and continuous variables
categorical_features <- c("homeType")
continuous_features <- c("bedrooms", "lotSize", "resoFacts.yearBuilt", "LivingArea...LotSize",
                         "relative_elevation", "distanceSchool", "beachProximity", "median_income",
                         "Crime Index", "School District Rating")

# Split the data (90/10 split)
set.seed(123)
train_index <- createDataPartition(df_model$price, p = 0.9, list = FALSE)
train_data <- df_model[train_index]
test_data <- df_model[-train_index]

# Prepare matrices for xgboost
dtrain <- xgb.DMatrix(data = as.matrix(train_data[, ..continuous_features]), label = train_data$price)
dtest <- xgb.DMatrix(data = as.matrix(test_data[, ..continuous_features]), label = test_data$price)

# Train the XGBoost model
params <- list(
  objective = "reg:squarederror",
  max_depth = 3,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.7,
  alpha = 10,
  lambda = 1,
  min_child_weight = 1,
  gamma = 0
)

xgboost_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 1000,
  watchlist = list(train = dtrain, test = dtest),
  print_every_n = 100
)

# Save the model
xgb.save(xgboost_model, "xgboost_house_model.model")

#APP----

# Load the trained model
xgboost_model <- xgb.load("xgboost_house_model.model")

# Define the UI
ui <- dashboardPage(
  dashboardHeader(title = "California Home Price Predictor"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Prediction", tabName = "prediction", icon = icon("dollar-sign")),
      menuItem("About", tabName = "about", icon = icon("info-circle"))
    )
  ),
  dashboardBody(
    tabItems(
      tabItem(
        tabName = "prediction",
        fluidRow(
          # Input Panel
          box(
            width = 4, title = "Input Parameters", status = "primary", solidHeader = TRUE,
            textInput("address", "Street Address", ""),
            textInput("city", "City", ""),
            textInput("zipcode", "ZIP Code", ""),
            numericInput("bedrooms", "Bedrooms", value = 3, min = 1),
            numericInput("lotSize", "Lot Size (sq ft)", value = 5000, min = 1),
            numericInput("yearBuilt", "Year Built", value = 2000, min = 1800),
            numericInput("livingAreaLotRatio", "Living Area / Lot Size", value = 0.5, min = 0, step = 0.01),
            numericInput("relativeElevation", "Relative Elevation", value = 10, min = -100, step = 1),
            numericInput("distanceSchool", "Distance to School (miles)", value = 1, min = 0, step = 0.1),
            numericInput("beachProximity", "Beach Proximity (miles)", value = 5, min = 0, step = 0.1),
            numericInput("schoolRating", "School District Rating", value = 7, min = 0, max = 10, step = 0.1),
            actionButton("predict", "Predict Price", class = "btn-primary btn-lg btn-block")
          ),
          # Output Panel
          column(
            width = 8,
            fluidRow(
              # Predicted Price Section
              box(
                width = NULL, title = "Predicted Price", status = "primary", solidHeader = TRUE,
                textOutput("predicted_price")
              )
            ),
            fluidRow(
              # Location Info Section
              box(
                width = NULL, title = "Location Info", status = "info", solidHeader = TRUE,
                uiOutput("crime_index_display"),
                uiOutput("median_income_display")
              )
            ),
            fluidRow(
              # Location Map Section
              box(
                width = NULL, title = "Location Map", status = "primary", solidHeader = TRUE,
                leafletOutput("map", height = 400)
              )
            )
          )
        )
      ),
      tabItem(
        tabName = "about",
        box(
          width = 12, title = "About This App", status = "info", solidHeader = TRUE,
          HTML("<p>This tool predicts house prices using an XGBoost model trained on real estate data. 
               Inputs such as bedrooms, lot size, year built, crime index, and median income are used for prediction.</p>")
        )
      )
    )
  )
)

# Define the server
server <- function(input, output, session) {
  
  # Reactive value for location data
  location_data <- reactive({
    req(input$address, input$city, input$zipcode)
    full_address <- paste(input$address, input$city, input$zipcode, "USA")
    result <- geo(address = full_address, method = "osm")  # Geocode using OpenStreetMap
    if (nrow(result) > 0 && !is.na(result$lat) && !is.na(result$long)) {
      list(lat = result$lat, lng = result$long, valid = TRUE)
    } else {
      list(valid = FALSE)
    }
  })
  
  # Fetch crime index based on city
  crime_index <- reactive({
    req(input$city)
    city_input <- trimws(tolower(input$city))
    crime_data$City <- trimws(tolower(crime_data$City))
    crime <- crime_data[crime_data$City == city_input, "Crime.Index"]
    if (length(crime) > 0) return(as.numeric(crime)) else return(NA)
  })
  
  # Fetch median income based on ZIP code
  median_income <- reactive({
    req(input$zipcode)
    
    # Convert input$zipcode to character
    zipcode_input <- as.character(trimws(input$zipcode))
    
    # Ensure zipcode_median_income is a data.table
    setDT(zipcode_median_income)
    
    # Filter for the matching ZIP code and pull the median income
    income <- zipcode_median_income[`address.zipcode` == zipcode_input, median_income]
    
    # Return the first value if found, otherwise NA
    if (length(income) > 0) {
      return(as.numeric(income[1]))
    } else {
      return(NA)
    }
  })
  
  # Predict price
  predicted_price <- eventReactive(input$predict, {
    loc <- location_data()
    if (!loc$valid) return("Address not found. Please check the input.")
    crime <- crime_index()
    income <- median_income()
    if (is.na(crime) || is.na(income)) return("Missing data: Crime Index or Median Income not found.")
    
    # Prepare input for prediction
    input_data <- data.table(
      bedrooms = input$bedrooms,
      lotSize = input$lotSize,
      `resoFacts.yearBuilt` = input$yearBuilt,
      `LivingArea...LotSize` = input$livingAreaLotRatio,
      relative_elevation = input$relativeElevation,
      distanceSchool = input$distanceSchool,
      beachProximity = input$beachProximity,
      median_income = as.numeric(income),
      `Crime Index` = as.numeric(crime),
      `School District Rating` = input$schoolRating
    )
    
    # Make prediction
    dmatrix <- xgb.DMatrix(data = as.matrix(input_data))
    prediction <- predict(xgboost_model, dmatrix)
    return(paste0("$", format(round(prediction, 2), big.mark = ",")))
  })
  
  # Render predicted price
  output$predicted_price <- renderText({
    predicted_price()
  })
  
  # Render crime index display
  output$crime_index_display <- renderUI({
    crime <- crime_index()
    if (is.na(crime)) {
      HTML("<p style='color: red;'>Crime Index not found for the entered city.</p>")
    } else {
      HTML(sprintf("<p><strong>Crime Index:</strong> %s</p>", format(crime, big.mark = ",")))
    }
  })
  
  # Render median income display
  output$median_income_display <- renderUI({
    income <- median_income()
    if (is.na(income)) {
      HTML("<p style='color: red;'>Median Income not found for the entered ZIP Code.</p>")
    } else {
      HTML(sprintf("<p><strong>Median Income:</strong> $%s</p>", format(income, big.mark = ",")))
    }
  })
  
  # Render map
  output$map <- renderLeaflet({
    loc <- location_data()
    if (loc$valid) {
      leaflet() %>%
        addTiles() %>%
        addMarkers(lat = loc$lat, lng = loc$lng, popup = paste(input$address, input$city, input$zipcode))
    } else {
      leaflet() %>%
        addTiles() %>%
        addPopups(0, 0, "Address not found. Please check the input.")
    }
  })
}

# Run the app
shinyApp(ui = ui, server = server)


