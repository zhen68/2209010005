import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图形样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")



# 读取数据
df = pd.read_csv('./Data5.csv')

print("数据集基本信息:")
print(f"数据形状: {df.shape}")
print(f"数据列: {list(df.columns)}")
print("\n前5行数据:")
print(df.head())

# 创建主要的6种可视化图形
fig = plt.figure(figsize=(20, 12))

# 1. 直方图 - 账单金额分布
plt.subplot(2, 3, 1)
plt.hist(df['total_bill'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
plt.title('1. 账单金额分布直方图', fontsize=14, fontweight='bold')
plt.xlabel('账单金额 ($)')
plt.ylabel('频次')
plt.grid(True, alpha=0.3)

# 2. 饼图 - 性别比例
plt.subplot(2, 3, 2)
gender_counts = df['sex'].value_counts()
colors = ['#ff9999', '#66b3ff']
plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', 
        colors=colors, startangle=90)
plt.title('2. 性别分布饼图', fontsize=14, fontweight='bold')

# 3. 散点图 - 账单金额与小费关系
plt.subplot(2, 3, 3)
plt.scatter(df['total_bill'], df['tip'], alpha=0.6, c=df['size'], 
           cmap='viridis', s=60)
plt.colorbar(label='聚餐人数')
plt.title('3. 账单金额与小费关系散点图', fontsize=14, fontweight='bold')
plt.xlabel('账单金额 ($)')
plt.ylabel('小费 ($)')
plt.grid(True, alpha=0.3)

# 4. 柱状图 - 不同日期的平均账单金额
plt.subplot(2, 3, 4)
day_avg = df.groupby('day')['total_bill'].mean()
bars = plt.bar(day_avg.index, day_avg.values, color=['#ffcc99', '#ff9999', '#99ccff', '#99ff99'])
plt.title('4. 不同日期平均账单金额柱状图', fontsize=14, fontweight='bold')
plt.xlabel('星期')
plt.ylabel('平均账单金额 ($)')
plt.xticks(rotation=45)
# 添加数值标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'${height:.1f}', ha='center', va='bottom')

# 5. 箱线图 - 不同用餐时间的小费分布
plt.subplot(2, 3, 5)
sns.boxplot(data=df, x='time', y='tip', palette='Set2')
plt.title('5. 不同用餐时间小费分布箱线图', fontsize=14, fontweight='bold')
plt.xlabel('用餐时间')
plt.ylabel('小费 ($)')

# 6. 条形图 - 吸烟者与非吸烟者的平均消费比较
plt.subplot(2, 3, 6)
smoker_data = df.groupby(['smoker', 'sex'])['total_bill'].mean().unstack()
smoker_data.plot(kind='bar', ax=plt.gca(), color=['#ff7f7f', '#7fbfff'])
plt.title('6. 吸烟者与非吸烟者平均消费比较', fontsize=14, fontweight='bold')
plt.xlabel('是否吸烟')
plt.ylabel('平均账单金额 ($)')
plt.legend(title='性别')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('餐厅数据可视化_主要6图.png', dpi=300, bbox_inches='tight')
# plt.show()

# 创建额外的高级可视化图形
fig2 = plt.figure(figsize=(20, 15))

# 7. 热力图 - 相关性矩阵
plt.subplot(2, 3, 1)
numeric_cols = ['total_bill', 'tip', 'size']
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
           square=True, fmt='.2f', cbar_kws={'label': '相关系数'})
plt.title('7. 数值变量相关性热力图', fontsize=14, fontweight='bold')

# 8. 小提琴图 - 不同聚餐人数的账单分布
plt.subplot(2, 3, 2)
sns.violinplot(data=df, x='size', y='total_bill', palette='pastel')
plt.title('8. 不同聚餐人数账单分布小提琴图', fontsize=14, fontweight='bold')
plt.xlabel('聚餐人数')
plt.ylabel('账单金额 ($)')

# 9. 气泡图 - 账单、小费、人数三维关系
plt.subplot(2, 3, 3)
bubble_sizes = df['size'] * 30
colors = df['total_bill']
scatter = plt.scatter(df['total_bill'], df['tip'], s=bubble_sizes, c=colors, 
                     cmap='YlOrRd', alpha=0.6, edgecolors='black', linewidth=0.5)
plt.colorbar(scatter, label='账单金额 ($)')
plt.title('9. 账单-小费-人数气泡图', fontsize=14, fontweight='bold')
plt.xlabel('账单金额 ($)')
plt.ylabel('小费 ($)')

# 10. 堆叠条形图 - 不同日期的性别分布
plt.subplot(2, 3, 4)
gender_day = pd.crosstab(df['day'], df['sex'])
gender_day.plot(kind='bar', stacked=True, ax=plt.gca(), 
               color=['#ff9999', '#66b3ff'])
plt.title('10. 不同日期性别分布堆叠条形图', fontsize=14, fontweight='bold')
plt.xlabel('星期')
plt.ylabel('人数')
plt.legend(title='性别')
plt.xticks(rotation=45)

# 11. 分组条形图 - 不同条件下的小费率
plt.subplot(2, 3, 5)
df['tip_rate'] = df['tip'] / df['total_bill'] * 100
tip_rate_data = df.groupby(['time', 'smoker'])['tip_rate'].mean().unstack()
tip_rate_data.plot(kind='bar', ax=plt.gca(), color=['#ffd700', '#ff6347'])
plt.title('11. 不同条件下小费率分组条形图', fontsize=14, fontweight='bold')
plt.xlabel('用餐时间')
plt.ylabel('小费率 (%)')
plt.legend(title='是否吸烟')
plt.xticks(rotation=45)

# 12. 密度图 - 账单金额和小费的双变量分布
plt.subplot(2, 3, 6)
sns.kdeplot(data=df, x='total_bill', y='tip', cmap='Blues', fill=True)
plt.scatter(df['total_bill'], df['tip'], alpha=0.3, s=20, color='red')
plt.title('12. 账单金额与小费密度分布图', fontsize=14, fontweight='bold')
plt.xlabel('账单金额 ($)')
plt.ylabel('小费 ($)')

plt.tight_layout()
plt.savefig('餐厅数据可视化_高级图形.png', dpi=300, bbox_inches='tight')
# plt.show()

# 创建雷达图（单独绘制，因为需要极坐标）
fig3, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# 准备雷达图数据
day_stats = df.groupby('day').agg({
    'total_bill': 'mean',
    'tip': 'mean', 
    'size': 'mean'
}).round(2)

# 归一化数据
scaler = MinMaxScaler()
day_stats_norm = pd.DataFrame(
    scaler.fit_transform(day_stats),
    columns=day_stats.columns,
    index=day_stats.index
)

# 绘制雷达图
angles = np.linspace(0, 2*np.pi, len(day_stats_norm.columns), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))

colors = ['red', 'blue', 'green', 'orange']
for i, day in enumerate(day_stats_norm.index):
    values = day_stats_norm.loc[day].values
    values = np.concatenate((values, [values[0]]))
    ax.plot(angles, values, 'o-', linewidth=2, label=day, color=colors[i])
    ax.fill(angles, values, alpha=0.25, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(['平均账单', '平均小费', '平均人数'])
ax.set_ylim(0, 1)
ax.set_title('13. 不同星期消费特征雷达图', fontsize=16, fontweight='bold', pad=30)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.savefig('餐厅数据_雷达图.png', dpi=300, bbox_inches='tight')
# plt.show()

# 数据统计分析
print("\n" + "="*50)
print("数据统计分析结果")
print("="*50)
print(f"总样本数: {len(df)}")
print(f"平均账单金额: ${df['total_bill'].mean():.2f}")
print(f"平均小费: ${df['tip'].mean():.2f}")
print(f"平均小费率: {(df['tip']/df['total_bill']*100).mean():.1f}%")
print(f"平均聚餐人数: {df['size'].mean():.1f}人")

print("\n分类变量分布:")
categorical_vars = ['sex', 'smoker', 'day', 'time']
for var in categorical_vars:
    print(f"\n{var}:")
    counts = df[var].value_counts()
    for category, count in counts.items():
        percentage = count/len(df)*100
        print(f"  {category}: {count}次 ({percentage:.1f}%)")

print("\n数值变量描述性统计:")
print(df[['total_bill', 'tip', 'size']].describe().round(2))

print("\n图形说明:")
print("1. 直方图 - 显示账单金额的分布情况")
print("2. 饼图 - 展示性别比例")
print("3. 散点图 - 分析账单金额与小费的关系，颜色表示聚餐人数")
print("4. 柱状图 - 比较不同星期的平均消费水平")
print("5. 箱线图 - 对比午餐和晚餐时间的小费分布")
print("6. 条形图 - 分析吸烟习惯和性别对消费的影响")
print("7. 热力图 - 显示数值变量间的相关性")
print("8. 小提琴图 - 展示不同聚餐人数的消费分布密度")
print("9. 气泡图 - 三维展示账单、小费、人数的关系")
print("10. 堆叠条形图 - 分析不同星期的性别构成")
print("11. 分组条形图 - 比较不同条件下的小费率")
print("12. 密度图 - 显示账单金额与小费的双变量密度分布")
print("13. 雷达图 - 多维度比较不同星期的消费特征")
