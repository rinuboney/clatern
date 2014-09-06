(ns clatern.implementations
  (:require [clatern.protocols :as cp]))

(def IMPLEMENTATIONS
  (array-map
   :knn 'clatern.knn
   :linear-regression 'clatern.linear-regression))

(defonce canonical-objects (atom {}))

(defn register-implementation
  [canonical-object]
  (swap! canonical-objects
         assoc
         (cp/implementation-key canonical-object)
         canonical-object))

(defn get-implementation-key [m]
  (cp/implementation-key m))

(defn try-load-implementation [k]
  (if-let [ns-sym (IMPLEMENTATIONS k)]
    (try
      (do
        (require ns-sym)
        (if (@canonical-objects k) :ok :warning-implementation-not-registered?))
      (catch Throwable t nil))))

(defn get-canonical-object [k]
  (if (try-load-implementation k)
      (@canonical-objects k)
      nil))
