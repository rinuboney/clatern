(ns clatern.io
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.dataset :refer :all]
            [clojure.data.csv :as csv]
            [clojure.java.io :as io]))

(defn- read-numbers [val]
  (let [n (read-string val)]
    (if (number? n) n val)))

(defn load-data [filename]
  (emap read-numbers (matrix (with-open [in-file (io/reader filename)]
                              (doall
                               (csv/read-csv in-file))))))

(defn load-dataset [filename]
  (let [data (load-data filename)]
    (dataset (first data) (rest data))))

(defn save-data [filename data]
  (with-open [out-file (io/writer filename)]
    (csv/write-csv out-file data)))

(defn save-dataset [filename data]
  (let [data (cons (:column-names data) (transpose (:columns data)))]
    (save-data filename data)))
