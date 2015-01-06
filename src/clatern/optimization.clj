(ns clatern.optimization
  (:require [clojure.core.matrix.operators :as M]))

;; Gradient Descent
;; ================
;;  X : input data
;;  y : target data
;;  init-theta : intial theta values
;;  alpha : learning rate
;;  lambda : regularization parameter
;;  num-iters : number of iterations

(defn gradient-descent [X y grad init-theta alpha lambda num-iters]
  (loop [i 0 theta init-theta]
    (if (= i num-iters)
      theta
      (recur (inc i)
             (M/- theta (M/* alpha (grad X y theta lambda)))))))

;; TODO: implement stochastic gradient descent
