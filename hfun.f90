subroutine hfun(x, p, v, h)

    double precision, intent(in)  :: x(2), p(2), v
    double precision, intent(out) :: h

    ! local variables
    double precision :: r, th, pr, pth
    double precision :: l, m

    r   = x(1)
    th  = x(2)

    pr  = p(1)
    pth = p(2)

    l   = 4d0/5d0
    m   = sqrt(sin(r)**2/(1d0-l*sin(r)**2))

    h   = v*pth + sqrt(pr**2 + (pth/m)**2)

end subroutine hfun