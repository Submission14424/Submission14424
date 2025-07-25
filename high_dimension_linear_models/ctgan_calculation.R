.libPaths("/mnt/ufs18/home-249/wangzun/anaconda3/envs/myr/lib/R/library")

library(data.table)
library(glmnet)  # 只保留必要的包
library(purrr)

# 文件路径配置（保持不变）
file_paths <- list(
  v1_file_path_source = "ctgan.csv",
  file_path_test = "test.csv"
)

read_source_data <- function(file_path) {
  return(read.csv(file_path, header = FALSE))
}

# 读取目标数据函数（保持不变）
read_target_data <- function(file_path) {
  target_data <- read.csv(file_path, header = FALSE)
  target_x <- as.matrix(target_data[, 1:511])
  target_y <- as.numeric(target_data[[512]])
  return(list(x = target_x, y = target_y))
}

# 读取数据（保持不变）
test_data <- read_target_data(file_paths$file_path_test)
v1_source_data <- read_source_data(file_paths$v1_file_path_source)

# 初始化参数（保持不变）

test_1x <- cbind(1, test_data$x)

# 获取任务ID（保持不变）
task_id <- as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))

# 切分比例配置（保持不变）
seq1 <- seq(from = 0.01, to = 0.3, by = 0.01)
seq2 <- seq(from = 0.32, to = 0.5, by = 0.02)
split_ratios <- c(seq1, seq2)
ratio <- split_ratios[task_id]


# 存储结果列表（简化）
pre_errors_v1 <- c()  # 预分配空间提升性能

# 修改后的100次循环
for (iter in 1:100) {
  set.seed(task_id + iter)  # 保持原有种子设置
  
  # 1. 随机抽样数据（保持原有抽样逻辑）
  v1_sub_data <- v1_source_data[sample(nrow(v1_source_data), floor(nrow(v1_source_data) * ratio)), ]
  
  # 2. 准备训练数据（移除批次划分）
  train_x <- as.matrix(v1_sub_data[, 1:511])
  train_y <- as.numeric(v1_sub_data[[512]])
  
  # 3. 使用glmnet进行Lasso回归（新增核心代码）
  fit_lasso <- cv.glmnet(x = train_x, y = train_y)
  y_pred <- predict(fit_lasso, newx = test_data$x)
  
  # 4. 存储预测误差（保持原有误差计算）
  pre_errors_v1[iter] <- mean((y_pred - test_data$y)^2)
}

# 结果保存（简化格式）
final_results <- data.frame(
  task_id = task_id,
  ratio = ratio,
  avg_pre_error_v1 = mean(pre_errors_v1)
)

# 保持原有保存逻辑
file_name <- paste0("100ctgan_", task_id, ".csv")
write.csv(final_results, file_name, row.names = FALSE)

cat("任务", task_id, "已完成并保存。\n")