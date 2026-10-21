rm(list = ls())

# load dplyr package for data handling
library(dplyr)

# set working directory (where data is located)
setwd("[NAMEOFYOURFOLDER]/data/")

# helper function for formatting numbers
fmt2 <- function(x){
  formatC(x, width = 2, format = "d", flag = "0")
}

# list all raw files in folder (each covering half a year)
all_file_names <- list.files(pattern = "Realisierter_Stromverbrauch")

# initialize empty data frame
dat_all <- data.frame()

# loop through files
for (file_name in all_file_names){
  # loda data
  dat_tmp1 <- read.csv(file_name,
                      header = TRUE, sep = ";")
  
  # tidy up data
  dat_tmp2 <- dat_tmp1 %>%
    transmute(date1 = .[,1], date2 = .[,2], 
              load = as.numeric(gsub("\\,", ".", gsub("\\.", "", .[,3])))/1000, # divide by 1000 to get GWh instead of MWh
              residual_load = as.numeric(gsub("\\,", ".", gsub("\\.", "", .[,6])))/1000) %>% # same for residual load
    mutate(year = as.integer(substr(date1, 7, 10)), # make calendar variables
           month = as.integer(substr(date1, 4, 5)),
           day = as.integer(substr(date1, 1, 2)), 
           hour = as.integer(substr(date1, 12, 13)) + 1, 
           date = as.Date(paste0(year, "-", fmt2(month), "-",
                                 fmt2(day))), 
           weekday = weekdays(date),
           comment = c(NA, diff(hour)), 
           comment = if_else(comment %in% c(0, 2), "time shift", "")) %>%
    select(date, year, month, day, hour, weekday, load, 
           residual_load, comment)
  
  # append to existing data
  dat_all <- rbind(dat_all, dat_tmp2)
}

# some checks
# neighboring rows should be 0 or 1 days apart
table(diff(dat_all$date))

# some hours are duplicated due to time shift
duplicated_hours <- duplicated(dat_all[, c("date", "hour")])
dat_all %>% filter(duplicated_hours)

# save processed data 
write.table(dat_all, "energy_data_2020_2026.csv", sep = ",", col.names = TRUE, 
            row.names = FALSE)
