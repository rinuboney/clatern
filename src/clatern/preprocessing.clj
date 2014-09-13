(ns clatern.preprocessing
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.dataset :refer :all]
            [clojure.core.matrix.operators :as M]
            [clojure.core.matrix.stats :refer :all]))

; API

(defmulti set-options (fn [model options coder] coder))
(defmulti fit (fn [model data coder] coder))
(defmulti encode (fn [model data coder] coder))

; Mean Normalization

(defmethod set-options :mean-normalization
  [model options coder]
  (assoc model :preprocessing (assoc (:preprocessing model) :mean-normalization options)))

(defmethod fit :mean-normalization
  [model data coder]
  (let [metadata (:mean-normalization (:preprocessing model))
        column-names (:column-names metadata)
        cols (transpose (:columns (select-columns data column-names)))
        mu (mean cols)
        std (sd cols)
        metadata (merge metadata {:mean mu, :sd std})]
    (assoc model :preprocessing (assoc (:preprocessing model) :mean-normalization metadata))))

(defn- mean-norm-arg [X mu sd]
  (M// (M/- X mu) sd))

(defn- mean-denorm-arg [X mu sd]
  (M/+ mu (M/* sd X)))

(defmethod encode :mean-normalization
  [model data coder]
  (let [metadata (:mean-normalization (:preprocessing model))
        col-names (:column-names metadata)
        cols (transpose (:columns (select-columns data col-names)))
        mu (:mean metadata)
        std (:sd metadata)
        normalized (matrix (map #(mean-norm-arg % mu std)
                                      (rows cols)))]
    (merge-datasets (except-columns data col-names)
                    (dataset col-names normalized))))
