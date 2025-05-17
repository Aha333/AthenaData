https://awadrahman.medium.com/recommended-python-libraries-for-practical-causal-ai-5642d718059d



# Uplift Model vs. CATE Model: What's the Difference?
Uplift模型更关注个体层面的增量效果，它的结果是关于个体的干预反应差异。
CATE模型则是在群体层面上估计不同条件下的干预效果，它的结果通常是某个群体（例如特定特征的个体）接受干预的平均效应。

Uplift一般需要A/B test数据， cate可能就是observation的regression on convariate了

# doubly robust estimation
下面这个例子给出了为什么 double robust， 
我有一个疑问： 是不是ATE是robust的，但是 CATE不是的呢？
https://matheusfacure.github.io/python-causality-handbook/12-Doubly-Robust-Estimation.html

# smoothing [我的理解]
the smoothing technique (i.e., regressing the individual treatment effects) 
can be applied to any CATE estimator, including S-learner, T-learner, X-learner, and Double Robust (DR). This method is essentially a way to refine and smooth the individual treatment effect estimates, 

S， T， X， DR都有对CATE的估计的公式。 方法都不太一样


# DR vs DML
https://zhuanlan.zhihu.com/p/626998556