(ns clatern.optimization
  (:require [clojure.core.matrix.operators :as M]))

(defn gradient-descent [X y grad init-theta alpha lambda num-iters]
  (loop [i 0 theta init-theta]
    (if (= i num-iters)
      theta
      (recur (inc i)
             (M/- theta (M/* alpha (grad X y theta lambda)))))))
