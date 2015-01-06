(ns clatern.naive_bayes
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.stats :refer :all]))

;; Gaussian Naive Bayes
;; ====================

(defn- stdv
  "return standard deviation replacing NaN and 0 with a minimum value"
  [values]
  (if (= (count values) 1)
    1e-9
    (emap #(max % 1e-9) (sd values))))

(defn- gaussian-prob
  "calculate probability in gaussian distribution"
  [x mean sd]
  (/ (exp (/ (- (pow (- x mean) 2)) (* 2 sd sd))) (sqrt (* 2 22/7 sd sd))))

(defn- class-probs
  "calculate probability of input for all classes"
  [v means sds priors]
  (let [classes (keys priors)]
     (into {}
       (for [c classes]
         [c (apply + (log (priors c)) (log (map gaussian-prob v (means c) (sds c))))]))))

(defn- predict
  "predict the class with highest probability"
  [v means sds priors]
  (let [probs (class-probs v means sds priors)]
    (first (apply max-key second probs))))

(defn- map-vals
  "Apply function f to all values in map m"
  [m f]
  (zipmap (keys m) (map f (vals m))))

(defn- calc-stats-per-class
  "calculate statistics of input data"
  [X y]
  (let [total (row-count X)
        seperated (group-by last (join-along 1 X y))
        priors (map-vals seperated #(/ (count %) total))
        seperated-y (map-vals seperated #(transpose (butlast (columns  %))))
        means (map-vals seperated-y mean)
        sds (map-vals seperated-y stdv)]
    [means sds priors]))

(defrecord GaussianNB [means sds priors]
  clojure.lang.IFn
  (invoke [this v] (predict v means sds priors))
  (applyTo [this args] (clojure.lang.AFn/applyToHelper this args)))

(defn gaussian-nb [X y]
  (let [[means sds priors] (calc-stats-per-class X y)]
    (GaussianNB. means sds priors)))


;; TODO: implement BernoulliNB and MultinomialNB
