(ns clatern.protocols)

(defprotocol Model
  (implementation-key [m])
  (set-options [m options])
  (fit [m data])
  (predict [m data]))
