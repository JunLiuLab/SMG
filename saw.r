#6.9(a)
set.seed(1)
n = 10000
D = c(); M = c(); weight = c()
for (j in 1:n) {
  v = 2*rbinom(30, 1, 0.5)-1
  w = 2*rbinom(30, 1, 0.5)-1
  for (i in 2:30) {
    v[i] = v[i] + v[i-1]
    w[i] = w[i] + w[i-1]
  }
  D[j] = abs(v[30]) + abs(w[30])
  M[j] = 0
  for (i in 1:30) M[j] = max(M[j], sum((w == w[i]) & (v == v[i])))
  weight[j] = exp(-(D[j] + sum((w == w[30]) & (v == v[30])))/2)
}
weight = weight / sum(weight)
c(sum(weight*D), sqrt(sum(weight*D^2) - sum(weight*D)^2)) # mean and standard deviation of D_30(x_30)

#6.9(b)
set.seed(1)
n = 10000
D = c(); M = c(); weight = c()
for (j in 1:n) {
  v = 2*rbinom(1,1,0.5)-1; w = 2*rbinom(1,1,0.5)-1
  weight[j] = 1
  for (i in 2:30) {
    v.minus = c(v, v[i-1] - 1); v.pluss = c(v, v[i-1] + 1)
    w.minus = c(w, w[i-1] - 1); w.pluss = c(w, w[i-1] + 1)
    weight.temp = c(
      exp(-(abs(v.minus[i]) + abs(w.minus[i]) + sum((v.minus == v.minus[i]) & (w.minus == w.minus[i])))/2),
      exp(-(abs(v.minus[i]) + abs(w.pluss[i]) + sum((v.minus == v.minus[i]) & (w.pluss == w.pluss[i])))/2),
      exp(-(abs(v.pluss[i]) + abs(w.minus[i]) + sum((v.pluss == v.pluss[i]) & (w.minus == w.minus[i])))/2),
      exp(-(abs(v.pluss[i]) + abs(w.pluss[i]) + sum((v.pluss == v.pluss[i]) & (w.pluss == w.pluss[i])))/2)
    )
    weight.temp = weight.temp / sum(weight.temp)
    choice = sample.int(4, 1, prob = weight.temp)
    if (choice == 1) {v = v.minus; w = w.minus; weight[j] = weight[j] / weight.temp[1]}
    if (choice == 2) {v = v.minus; w = w.pluss; weight[j] = weight[j] / weight.temp[2]}
    if (choice == 3) {v = v.pluss; w = w.minus; weight[j] = weight[j] / weight.temp[3]}
    if (choice == 4) {v = v.pluss; w = w.pluss; weight[j] = weight[j] / weight.temp[4]}
  }
  D[j] = abs(v[30]) + abs(w[30])
  M[j] = 0
  for (i in 1:30) M[j] = max(M[j], sum((w == w[i]) & (v == v[i])))
  weight[j] = weight[j] * exp(-(D[j] + sum((v == v[30]) & (w == w[30])))/2)
}
weight = weight / sum(weight)
c(sum(weight*D), sqrt(sum(weight*D^2) - sum(weight*D)^2)) # mean and standard deviation of D_30 (x_30)

#6.9(c)
set.seed(1)
n = 10000
D = c(); M = c(); weight = c()
for (j in 1:n) {
  v = 0
  w = 0
  weight[j] = 1
  for (i in 2:31) {
    v.minus = c(v, v[i-1] - 1); v.pluss = c(v, v[i-1] + 1)
    w.minus = c(w, w[i-1] - 1); w.pluss = c(w, w[i-1] + 1)
    weight.temp = c(
      sum((v == v.minus[i]) & (w == w.minus[i])),
      sum((v == v.minus[i]) & (w == w.pluss[i])),
      sum((v == v.pluss[i]) & (w == w.minus[i])),
      sum((v == v.pluss[i]) & (w == w.pluss[i]))
    )
    for (k in 1:4) {
      if (weight.temp[k] == 0) weight.temp[k] = 1
      else weight.temp[k] = 0
    }
    if (sum(weight.temp) == 0) break
    weight.temp = weight.temp / sum(weight.temp)
    choice = sample.int(4, 1, prob = weight.temp)
    if (choice == 1) {v = v.minus; w = w.minus; weight[j] = weight[j] / weight.temp[1]}
    if (choice == 2) {v = v.minus; w = w.pluss; weight[j] = weight[j] / weight.temp[2]}
    if (choice == 3) {v = v.pluss; w = w.minus; weight[j] = weight[j] / weight.temp[3]}
    if (choice == 4) {v = v.pluss; w = w.pluss; weight[j] = weight[j] / weight.temp[4]}
  }
  if (sum(weight.temp) == 0) {
    D[j] = 0
    M[j] = 0
    weight[j] = 0
    next
  }
  D[j] = abs(v[31]) + abs(w[31])
  M[j] = 0
  for (i in 1:31) M[j] = max(M[j], sum((w == w[i]) & (v == v[i])))
}
weight = weight / sum(weight)
c(sum(weight*D), sqrt(sum(weight*D^2) - sum(weight*D)^2)) # mean and standard deviation of D_30 (x_30)

#6.9(d)
set.seed(1)
n = 1000
ts = c(2, 3, 5, 8, 11, 15, 20)
eff = c()
for (k in 1:length(ts)) {
  t = ts[k]
  D = c(); M = c(); weight = c()
  for (j in 1:n) {
    v = 0
    w = 0
    for (i in 2:t) {
      v[i] = v[i-1]+2*rbinom(1, 1, 0.5)-1
      w[i] = w[i-1]+2*rbinom(1, 1, 0.5)-1
    }
    D[j] = abs(v[t]) + abs(w[t])
    M[j] = 0
    for (i in 1:t) M[j] = max(M[j], sum((w == w[i]) & (v == v[i])))
    if (M[j] > 1) weight[j] = 0 else weight[j] = 1
  }
  if (sum(weight) > 0) {
    weight = weight / sum(weight)
    # c(sum(weight*D), sqrt(sum(weight*D^2) - sum(weight*D)^2)) # mean and standard deviation of D_30 (x_30)
    # c(sum(weight*M), sqrt(sum(weight*M^2) - sum(weight*M)^2)) # mean and standard deviation of M_30 (x_{1:30})
    # 1/sum(weight^2) # effective sample size
    eff[k] = 1/sum(weight^2)/n
  } else {
    eff[k] = 0
  }
}
data.frame(T=ts,efficiency=eff)