(ns clatern.utils)

(defn map-values
   "Apply function f to all values in map m"
    [f m]
    (zipmap (keys m) (map f (vals m))))
