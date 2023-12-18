set.seed(1234567890)
library(geosphere)

stations <- read.csv("stations.csv", fileEncoding = "latin1")
temps <- read.csv("temps50k.csv")

# Ensure that air_temperature is numeric, and merge the datasets on station_number
temps$air_temperature <- as.numeric(temps$air_temperature)
st <- merge(stations, temps, by = "station_number")

# Filter out the data for the forecast date and time
st$date <- as.Date(st$date)
forecast_date <- as.Date("2013-11-04")
st <- st[st$date <= forecast_date, ]

# Define the forecast times
forecast_times <- seq(from = as.POSIXct("04:00:00", format = "%H:%M:%S"),
                      to = as.POSIXct("24:00:00", format = "%H:%M:%S"),
                      by = "2 hours")

# define parameters for the kernels
h_distance <- 123456  
h_date <- 10          
h_time <- 6           
a <- 58.4274
b <- 14.826


# Gaussian kernel function
gaussian_kernel <- function(x, h) {
  (1 / (h * sqrt(2 * pi))) * exp(-0.5 * (x / h)^2)
}

# Initialize vectors for storing predictions
temps_sum <- vector("numeric", length(forecast_times))
temps_product <- vector("numeric", length(forecast_times))

# Initialize vectors for storing kernel values for plotting
distance_kernels_plot <- vector("list", length(forecast_times))

# Prediction loop
for (i in seq_along(forecast_times)) {
	  # Filter out temperatures with a date later than the forecast date
	  st_filtered <- st[st$date < forecast_date | (st$date == forecast_date & as.POSIXct(st$time, format = "%H:%M:%S") <= forecast_times[i]), ]
	  
	  # Calculate the distance kernel
	  st_filtered$distance <- distHaversine(matrix(c(st_filtered$longitude, st_filtered$latitude), ncol = 2), c(b, a))
	  distance_kernel <- gaussian_kernel(st_filtered$distance, h_distance)
	  
	  # Calculate the date kernel
	  date_diff <- as.numeric(forecast_date - st_filtered$date)
	  date_kernel <- gaussian_kernel(date_diff, h_date)
	  
	  # Calculate the time kernel
	  time_diff <- abs(as.numeric(difftime(forecast_times[i], as.POSIXct(st_filtered$time, format = "%H:%M:%S"), units = "hours")))
	  time_kernel <- gaussian_kernel(time_diff, h_time)


		# Calculate and store the kernels for plotting
	  distance_kernels_plot[[i]] <- gaussian_kernel(st_filtered$distance, h_distance)
	  
	  # Combine the kernels (sum and product)
	  combined_kernel_sum <- distance_kernel + date_kernel + time_kernel
	  combined_kernel_product <- distance_kernel * date_kernel * time_kernel
	  
 	 # Compute predictions
    temps_sum[i] <- sum(st_filtered$air_temperature * combined_kernel_sum) / sum(combined_kernel_sum)
    temps_product[i] <- sum(st_filtered$air_temperature * combined_kernel_product) / sum(combined_kernel_product)

}

# Define the times for plotting
times <- seq(from = 4, to = 24, by = 2)

st_for_plotting <- st[st$date < forecast_date | (st$date == forecast_date & as.POSIXct(st$time, format = "%H:%M:%S") <= forecast_times[1]), ]

distance_for_plotting <- distHaversine(matrix(c(st_for_plotting$longitude, st_for_plotting$latitude), ncol = 2),
                                       c(b, a))
distance_kernel_for_plotting <- gaussian_kernel(distance_for_plotting, h_distance)

# Plot the sub-kernels and combined kernels
plot(distance_for_plotting, distance_kernel_for_plotting, type = "o", 
     xlab = "Distance", ylab = "Kernel Value", main = "Distance Kernel")

plot(times, temps_sum, type = "o", 
     xlab = "Time", ylab = "Predicted Temperature", main = "Summed Kernels")

plot(times, temps_product, type = "o", 
     xlab = "Time", ylab = "Predicted Temperature", main = "Product Kernels")


comparison_table <- data.frame(
  Time = format(forecast_times, "%H:%M"),
  Summed_Kernel_Prediction = temps_sum,
  Product_Kernel_Prediction = temps_product,
  Difference = temps_sum - temps_product
)

# View the comparison table
print(comparison_table)
