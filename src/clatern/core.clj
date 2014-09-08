(ns clatern.core
  (:require [clatern.protocols :as cp]
            [clatern.implementations :as imp]))

(defn new-model [key]
  (imp/get-canonical-object key))

(defn set-options [m options]
  (cp/set-options m options))

(defn fit [m data]
  (cp/fit m data))

(defn predict [m data]
  (cp/predict m data))
