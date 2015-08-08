(ns clatern.protocols)

(defprotocol ClassProbabilityEstimator
  "Protocol for classifiers that can estimate
   class probabilities for a sample vector."  
  (predict-prob [this v] "Predict class probabilities for `v`.")
  (predict-log-prob [this v] "Predict class log probabilities for `v`."))
