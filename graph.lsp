(defun graph ( pts sls tls )

    (   (lambda ( l )
            (foreach x l (text (cdr x) (itoa (car x)) 0.0 1))
            (mapcar
               '(lambda ( a b / p q r )
                    (setq p (cdr (assoc a l))
                          q (cdr (assoc b l))
                          r (angle p q)
                    )
                    (entmake (list '(0 . "LINE") (cons 10 p) (cons 11 q) '(62 . 8)))
                    (text
                        (mapcar '(lambda ( x y ) (/ (+ x y) 2.0)) p q)
                        (roundupto (distance p q) 12.4)
                        (if (and (< (* pi 0.5) r) (<= r (* pi 1.5))) (+ r pi) r)
                        2
                    )
                )
                sls tls
            )
        )
        (mapcar 'cons (vl-sort (append sls tls) '<) pts)
    )
)
(defun text ( p s a c )
    (entmake
        (list
           '(0 . "TEXT")
            (cons 10 p)
            (cons 11 p)
            (cons 50 a)
            (cons 01 s)
            (cons 62 c)
           '(40 . 2)
           '(72 . 1)
           '(73 . 2)
        )
    )
)

(defun roundupto ( x m / d r )
    (setq d (getvar 'dimzin))
    (setvar 'dimzin 8)
    (setq r (rtos (* m (fix (+ 1 -1e-8 (/ x (float m))))) 2 8))
    (setvar 'dimzin d)
    r
)