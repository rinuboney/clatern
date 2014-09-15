(ns clatern.preprocessing
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.dataset :refer :all]
            [clojure.core.matrix.operators :as M]
            [clojure.core.matrix.stats :refer :all]))

;; API

(defmulti fit (fn [model data coder] coder))
(defmulti encode (fn [model data coder] coder))
(defmulti decode (fn [model data coder] coder))

(defn set-options
  [model options coder]
  (assoc model :preprocessing (assoc (:preprocessing model) coder options)))

;; Mean Normalization
(defmethod fit :mean-normalization
  [model data coder]
  (let [metadata (:mean-normalization (:preprocessing model))
        column-names (:column-names metadata)
        cols (transpose (:columns (select-columns data column-names)))
        mu (mean cols)
        std (sd cols)
        metadata (merge metadata {:mean mu, :sd std})]
    (assoc model :preprocessing (assoc (:preprocessing model) coder metadata))))

(defn- mean-norm-arg [X mu sd]
  (M// (M/- X mu) sd))

(defn- de-mean-norm-arg [X mu sd]
  (M/+ mu (M/* X sd)))

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

(defmethod decode :mean-normalization
  [model data coder]
  (let [metadata (:mean-normalization (:preprocessing model))
        col-names (:column-names metadata)
        cols (transpose (:columns (select-columns data col-names)))
        mu (:mean metadata)
        std (:sd metadata)
        denormalized (matrix (map #(de-mean-norm-arg % mu std)
                                      (rows cols)))]
    (merge-datasets (except-columns data col-names)
                    (dataset col-names denormalized))))

;; Min Max Scaling

(defn- min-max-scale [X min max]
  (M// (M/- X min) (M/- max min)))

(defn- de-min-max-scale [X min max]
  (M/+ min (M/* X (M/- max min))))

(defmethod fit :min-max-scaling
  [model data coder]
  (let [metadata (:min-max-scaling (:preprocessing model))
        column-names (:column-names metadata)
        cols (transpose (:columns (select-columns data column-names)))
        cmin (or (:min metadata) (map #(apply min %) (columns cols)))
        cmax (or (:max metadata) (map #(apply max %) (columns cols)))
        metadata (merge metadata {:min cmin, :max cmax})]
    (assoc model :preprocessing (assoc (:preprocessing model) coder metadata))))

(defmethod encode :min-max-scaling
  [model data coder]
  (let [metadata (:min-max-scaling (:preprocessing model))
        column-names (:column-names metadata)
        cols (transpose (:columns (select-columns data column-names)))
        cmin (:min metadata)
        cmax (:max metadata)
        scaled (matrix (map #(min-max-scale % cmin cmax) (rows cols)))]
    (merge-datasets (except-columns data column-names)
                    (dataset column-names scaled))))

(defmethod decode :min-max-scaling
  [model data coder]
  (let [metadata (:min-max-scaling (:preprocessing model))
        column-names (:column-names metadata)
        cols (transpose (:columns (select-columns data column-names)))
        cmin (:min metadata)
        cmax (:max metadata)
        descaled (matrix (map #(de-min-max-scale % cmin cmax) (rows cols)))]
    (merge-datasets (except-columns data column-names)
                    (dataset column-names descaled))))

;; Label Encoding

(defmethod fit :label-encoding
  [model data coder]
  (let [metadata (coder (:preprocessing model))
        column-names (:column-names metadata)
        cols (transpose (:columns (select-columns data column-names)))
        labels (map distinct (columns cols))
        metadata (merge metadata {:labels labels})]
    (assoc model :preprocessing (assoc (:preprocessing model) coder metadata))))

(defmethod encode :label-encoding
  [model data coder]
  (let [metadata (coder (:preprocessing model))
        column-names (:column-names metadata)
        cols (:columns (select-columns data column-names))
        labels (:labels metadata)
        coded (for [i (range (count column-names))]
                (let [col (nth cols i)
                      ilabels (nth labels i)]
                  (map #(.indexOf ilabels %) col)))
        coded (transpose coded)]
    (merge-datasets (except-columns data column-names)
                    (dataset column-names coded))))

(defmethod decode :label-encoding
  [model data coder]
  (let [metadata (coder (:preprocessing model))
        column-names (:column-names metadata)
        cols (:columns (select-columns data column-names))
        labels (:labels metadata)
        decoded (for [i (range (count column-names))]
                (let [col (nth cols i)
                       ilabels (nth labels i)]
                  (map #(nth ilabels %) col)))
        decoded (transpose decoded)]
    (merge-datasets (except-columns data column-names)
                    (dataset column-names decoded))))
