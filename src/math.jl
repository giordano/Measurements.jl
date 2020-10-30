### math.jl
#
# Copyright (C) 2016, 2017 Mosè Giordano.
#
# Maintainer: Mosè Giordano <mose AT gnu DOT org>
# Keywords: uncertainty, error propagation, physics
#
# This file is a part of Measurements.jl.
#
# License is MIT "Expat".
#
### Commentary:
#
# This file contains definition of mathematical functions that support
# Measurement objects.
#
# Note: some functions defined here (like all degree-related and reciprocal
# trigonometric functions, fld, cld, hypot, cbrt, abs, mod) are redundant in the
# sense that you would get the correct result also without their definitions,
# but having them defined here avoids some calculations and slightly improves
# performance.  Likewise, multiple methods are provided for functions taking two
# (or more) arguments because when only one argument is of Measurement type we
# can use the simple `result' function for one derivative that is faster than
# the generic method.
#
### Code:

# FiniteDifferences is used to calculate numerical derivatives in "@uncertain" macro.
using FiniteDifferences

export @uncertain

# This function is to be used by methods of mathematical operations to produce a
# `Measurement' object in output.  Arguments are:
#   * val: the nominal result of operation G(a)
#   * der: the derivative ∂G/∂a of G with respect to the variable a
#   * a: the only argument of G
# In this simple case of unary function, we don't have the problem of correlated
# variables (thus making this method much faster than the next one), so we can
# calculate the uncertainty of G(a) as
#   σ_G = |σ_a·∂G/∂a|
# The list of derivatives with respect to each measurement is updated with
#   ∂G/∂a · previous_derivatives
@inline function result(val::T, der::Real, a::Measurement{<:AbstractFloat}) where {T<:Real}
    newder = empty_der1(a)
    @inbounds for tag in keys(a.der)
        if ! iszero(tag[2]) # Skip values with 0 uncertainty
            newder = Derivatives(newder, tag=>der*a.der[tag])
        end
    end
    # If uncertainty of "a" is null, the uncertainty of result is null as well,
    # even if the derivative is NaN or infinite.  In any other case, use
    # σ_G = |σ_a·∂G/∂a|.
    σ = iszero(a.err) ? a.err : abs(der*a.err)
    # The tag for derived quantities is 0, for independent ones tag > 0.
    Measurement{float(T)}(val, σ, UInt64(0), newder)
end

# Get the common type parameter of a collection of Measurement objects.  The first two
# methods are for the trivial cases of homogeneous tuples and arrays, the last, inefficient,
# method is for inhomogeneous collections (probably the least common case).
gettype(::Tuple{Vararg{Measurement{T}}}) where {T<:AbstractFloat} = T
gettype(::AbstractArray{Measurement{T}}) where {T<:AbstractFloat} = T
_eltype(::Measurement{T}) where {T<:AbstractFloat} = T
gettype(collection) = promote_type(_eltype.(collection)...)

# This function is similar to the previous one, but applies to mathematical
# operations with more than one argument, so the formula to propagate
# uncertainty is more complicated because we have to take into account
# correlation between arguments.  The arguments are the same as above, but `der'
# and `a' are tuples of the same length (`der' has the derivatives of G with
# respect to the corresponding variable in `a').
#
# Suppose we have a function G = G(a1, a2) of two arguments.  a1 and a2 are
# correlated, because they come from some mathematical operations on really
# independent variables x, y, z, say a1 = a1(x, y), a2 = a2(x, z).  The
# uncertainty on G(a1, a2) is calculated as follows:
#   σ_G = sqrt((σ_x·∂G/∂x)^2 + (σ_y·∂G/∂y)^2 + (σ_z·∂G/∂z)^2)
# where ∂G/∂x is the partial derivative of G with respect to x, and so on.  We
# can expand the previous formula to:
#   σ_G = sqrt((σ_x·(∂G/∂a1·∂a1/∂x + ∂G/∂a2·∂a2/∂x))^2 + (σ_y·∂G/∂a1·∂a1/∂y)^2 +
#               + (σ_z·∂G/∂a2·∂a2/∂z)^2)
@inline function result(val, der, a)
    @assert length(der) == length(a)
    T = gettype(a)
    nil::T = zero(T)
    err = nil
    newder = empty_der2(nil)
    # Iterate over all independent variables.  We first iterate over all
    # variables listed in `a' in order to get all independent variables upon
    # which those variables depend, then we get the `tag' of each independent
    # variable, skipping variables that have been already taken into account.
    @inbounds for y in a
        for tag in keys(y.der)
            if tag ∉ keys(newder) # Skip independent variables already considered
                σ_x = tag[2]
                if ! iszero(σ_x)  # Skip values with 0 uncertainty
                    ∂G_∂x::T = nil
                    # Iteratate over all the arguments of the function
                    for (i, x) in enumerate(a)
                        # Calculate the derivative of G with respect to the
                        # current independent variable.  In the case of the x
                        # independent variable of the example above, we should
                        # get   ∂G/∂x = ∂G/∂a1·∂a1/∂x + ∂G/∂a2·∂a2/∂x
                        ∂a_∂x = derivative(x, tag) # ∂a_i/∂x
                        if ! iszero(∂a_∂x) # Skip values with 0 partial derivative
                            # der[i] = ∂G/∂a_i
                            ∂G_∂x = ∂G_∂x + der[i]*∂a_∂x
                        end
                    end
                    if ! iszero(∂G_∂x)
                        # Add (σ_x·∂G/∂x)^2 to the total uncertainty (squared), but only if
                        # the derivative is not zero.
                        newder = Derivatives(newder, tag=>∂G_∂x)
                        err = err + abs2(σ_x*∂G_∂x)
                    end
                end
            end
        end
    end
    return Measurement(T(val), sqrt(err), UInt64(0), newder)
end

# "result" function for complex-valued functions of one real argument (like "besselh").
# This takes the same argument as the first implementation of "result", but with complex
# "val" and "der".
result(val::Complex, der::Complex, a::Measurement) =
    complex(result(real(val), real(der), a), result(imag(val), imag(der), a))

# "result" function for complex-valued functions of one complex argument:
#
#   f(z) = f(x, y) = u(x, y) + im * v(x, y)
#
# where z = x + im * y, x and y are real variables, u and v are real-valued functions.
# Arguments of this methods are:
#
#   1) the nominal value of the result (a complex number)
#   2) the 4-tuple of derivatives (∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y)
#   3) the input measurement
result(val::Complex, der, a::Complex{<:Measurement}) =
    complex(result(real(val), der[1:2], reim(a)),
            result(imag(val), der[3:4], reim(a)))

### @uncertain macro.
"""
    @uncertain f(value ± stddev, ...)

A macro to calculate `f(value) ± uncertainty`, with `uncertainty` derived from `stddev`
according to rules of linear error propagation theory.

Function `f` can accept any number of real arguments.
"""
macro uncertain(expr::Expr, p::Int=4)
    f = esc(expr.args[1]) # Function name
    n = length(expr.args) - 1
    if n == 1
        a = esc(expr.args[2]) # Argument, of Measurement type
        return quote
            let x = measurement($a)
                result($f(x.val), FiniteDifferences.central_fdm($(esc(p)), 1)($f, x.val), x)
            end
        end
    else
        a = expr.args[2:end] # Arguments, as an array of expressions
        args = :([])  # Build up array of arguments
        [push!(args.args, :(measurement($(esc(a[i]))))) for i=1:n] # Fill the array
        argsval =:([])  # Build up the array of values of arguments
        [push!(argsval.args, :($(args.args[i]).val)) for i=1:n] # Fill the array
        return :( result($f($argsval...),
                         FiniteDifferences.grad(FiniteDifferences.central_fdm($(esc(p)), 1),
                                                x -> $f(x...), $argsval)[1],
                         $args) )
    end
end

### Elementary arithmetic operations:

# Addition: +
Base.:+(a::Measurement, b::Measurement) = result(a.val + b.val, (1, 1), (a, b))
Base.:+(a::Real, b::Measurement) = result(a + b.val, 1, b)
Base.:+(a::Measurement, b::Bool) = result(a.val + b, 1, a)
Base.:+(a::Measurement, b::Real) = result(a.val + b, 1, a)

# Subtraction: -
Base.:-(a::Measurement) = result(-a.val, -1, a)
Base.:-(a::Measurement, b::Measurement) = result(a.val - b.val, (1, -1), (a, b))
Base.:-(a::Real, b::Measurement) = result(a - b.val, -1, b)
Base.:-(a::Measurement, b::Real) = result(a.val - b, 1, a)

# Multiplication: *
function Base.:*(a::Measurement, b::Measurement)
    aval = a.val
    bval = b.val
    return result(aval*bval, (bval, aval), (a, b))
end
Base.:*(a::Bool, b::Measurement) = result(a*b.val, a, b)
Base.:*(a::Real, b::Measurement) = result(a*b.val, a, b)
Base.:*(a::Measurement, b::Bool) = result(a.val*b, b, a)
Base.:*(a::Measurement, b::Real) = result(a.val*b, b, a)

# muladd and fma

for f in (:fma, :muladd)
    @eval begin
        # All three arguments are Measurement
        function Base.$f(a::Measurement, b::Measurement, c::Measurement)
            x = a.val
            y = b.val
            z = c.val
            return result(($f)(x, y, z), (y, x, one(z)), (a, b, c))
        end

        # First argument is always Measurement
        function Base.$f(a::Measurement, b::Measurement, c::Real)
            x = a.val
            y = b.val
            return result(($f)(x, y, c), (y, x), (a, b))
        end

        function Base.$f(a::Measurement, b::Real, c::Measurement)
            x = a.val
            z = c.val
            return result(($f)(x, b, z), (b, one(z)), (a, c))
        end

        Base.$f(a::Measurement, b::Real, c::Real) =
            result(($f)(a.val, b, c), b, a)

        # Secon argument is always Measurement
        function Base.$f(a::Real, b::Measurement, c::Measurement)
            y = b.val
            z = c.val
            return result(($f)(a, y, z), (a, one(z)), (b, c))
        end

        Base.$f(a::Real, b::Measurement, c::Real) =
            result(($f)(a, b.val, c), a, b)

        # Third argument is Measurement
        function Base.$f(a::Real, b::Real, c::Measurement)
            z = c.val
            return result(($f)(a, b, z), one(z), c)
        end
    end
end

# Division: /, div, fld, cld
function Base.:/(a::Measurement, b::Measurement)
    x = a.val
    y = b.val
    oneovery = 1 / y
    return result(x / y, (oneovery, -x * abs2(oneovery)), (a, b))
end
Base.:/(a::Real, b::Measurement) = result(a/b.val, -a/abs2(b.val), b)
Base.:/(a::Measurement{T}, b::Real) where {T<:AbstractFloat} = result(a.val/b, 1/T(b), a)

# 0.0 as partial derivative for both arguments of "div", "fld", "cld" should be
# correct for most cases.  This has been tested against "@uncertain" macro.
Base.div(a::Measurement, b::Measurement) = result(div(a.val, b.val), (0, 0), (a, b))
Base.div(a::Measurement, b::Real) = result(div(a.val, b), 0, a)
Base.div(a::Real, b::Measurement) = result(div(a, b.val), 0, b)

Base.fld(a::Measurement, b::Measurement) = result(fld(a.val, b.val), (0, 0), (a, b))
Base.fld(a::Measurement, b::Real) = result(fld(a.val, b), 0, a)
Base.fld(a::Real, b::Measurement) = result(fld(a, b.val), 0, b)

Base.cld(a::Measurement, b::Measurement) = result(cld(a.val, b.val), (0, 0), (a, b))
Base.cld(a::Measurement, b::Real) = result(cld(a.val, b), 0, a)
Base.cld(a::Real, b::Measurement) = result(cld(a, b.val), 0, b)

# Inverse: inv
function Base.inv(a::Measurement)
    inverse = inv(a.val)
    return result(inverse, -abs2(inverse), a)
end

# signbit

Base.signbit(a::Measurement) = signbit(a.val)

# Power: ^

function Base.:^(a::Measurement, b::Measurement)
    aval = a.val
    bval = b.val
    pow = aval^bval
    return result(pow, (aval^(bval - 1)*bval, pow*log(aval)), (a, b))
end

function Base.:^(a::Measurement, b::Integer)
    x = a.val
    return result(x ^ b, b * x ^ (b - 1), a)
end

function Base.:^(a::Measurement{T},  b::Rational) where {T<:AbstractFloat}
    x = a.val
    return result(x ^ b, b * x ^ (b - one(T)), a)
end

function Base.:^(a::Measurement,  b::Real)
    x = a.val
    return result(x ^ b, b * x ^ (b - 1), a)
end

Base.:^(::Irrational{:ℯ}, b::Measurement) = exp(b)

function Base.:^(a::Real,  b::Measurement)
    res = a^b.val
    return result(res, res*log(a), b)
end

function Base.exp2(a::Measurement{T}) where {T<:AbstractFloat}
    pow = exp2(a.val)
    return result(pow, pow*log(T(2)), a)
end

### Trigonometric functions

# deg2rad, rad2deg

Base.deg2rad(a::Measurement) = a * (oftype(a.val, pi) / 180)
Base.rad2deg(a::Measurement) = a * (180 / oftype(a.val, pi))

# Cosine: cos, cosd, cosh

function Base.cos(a::Measurement)
    s, c = sincos(a.val)
    return result(c, -s, a)
end

function Base.cosd(a::Measurement)
    aval = a.val
    return result(cosd(aval), -deg2rad(sind(aval)), a)
end

function Base.cosh(a::Measurement)
    aval = a.val
    result(cosh(aval), sinh(aval), a)
end

# Sine: sin, sind, sinh

function Base.sin(a::Measurement)
    s, c = sincos(a.val)
    return result(s, c, a)
end

function Base.sind(a::Measurement)
    aval = a.val
    return result(sind(aval), deg2rad(cosd(aval)), a)
end

function Base.sinh(a::Measurement)
    aval = a.val
    result(sinh(aval), cosh(aval), a)
end

# Sincos: sincos

function Base.sincos(a::Measurement)
    s, c = sincos(a.val)
    return (result(s, c, a), result(c, -s, a))
end

if isdefined(Base, :sincospi)
    function Base.sincospi(a::Measurement)
        s, c = sincospi(a.val)
        return (result(s, c * pi, a), result(c, -s * pi, a))
    end
end

# Tangent: tan, tand, tanh

function Base.tan(a::Measurement)
    aval = a.val
    return result(tan(aval), abs2(sec(aval)), a)
end

function Base.tand(a::Measurement)
    aval = a.val
    return result(tand(aval), deg2rad(abs2(secd(aval))), a)
end

function Base.tanh(a::Measurement)
    aval = a.val
    return result(tanh(aval), abs2(sech(aval)), a)
end

# Other trig-related functions: sinpi, cospi, sinc, cosc

function Base.sinpi(a::Measurement)
    x = a.val
    return result(sinpi(x), cospi(x) * pi, a)
end

function Base.cospi(a::Measurement)
    x = a.val
    return result(cospi(x), -sinpi(x) * pi, a)
end

function Base.sinc(a::Measurement)
    x = a.val
    return result(sinc(x), cosc(x), a)
end

function Base.cosc(a::Measurement)
    x = a.val
    res = cosc(x)
    return result(res,
                  iszero(x) ? -oftype(x, pi) ^ 2 / 3 : -2 * res / x - sinc(x) * oftype(x, pi) ^ 2,
                  a)
end

# Inverse trig functions: acos, acosd, acosh, asin, asind, asinh, atan, atand, atanh,
#                         asec, acsc, acot, asech, acsch, acoth

function Base.acos(a::Measurement{T}) where {T<:AbstractFloat}
    aval = a.val
    return result(acos(aval), -inv(sqrt(one(T) - abs2(aval))), a)
end

function Base.acosd(a::Measurement{T}) where {T<:AbstractFloat}
    aval = a.val
    return result(acosd(aval), -rad2deg(inv(sqrt(one(T) - abs2(aval)))), a)
end

function Base.acosh(a::Measurement{T}) where {T<:AbstractFloat}
    aval = a.val
    return result(acosh(aval), inv(sqrt(abs2(aval) - one(T))), a)
end

function Base.asin(a::Measurement{T}) where {T<:AbstractFloat}
    aval = a.val
    return result(asin(aval), inv(sqrt(one(T) - abs2(aval))), a)
end

function Base.asind(a::Measurement{T}) where {T<:AbstractFloat}
    aval = a.val
    return result(asind(aval), rad2deg(inv(sqrt(one(T) - abs2(aval)))), a)
end

function Base.asinh(a::Measurement{T}) where {T<:AbstractFloat}
    aval = a.val
    return result(asinh(aval), inv(hypot(aval, one(T))), a)
end

function Base.atan(a::Measurement{T}) where {T<:AbstractFloat}
    aval = a.val
    return result(atan(aval), inv(abs2(aval) + one(T)), a)
end

function Base.atand(a::Measurement{T}) where {T<:AbstractFloat}
    aval = a.val
    return result(atand(aval), rad2deg(inv(abs2(aval) + one(T))), a)
end

function Base.atanh(a::Measurement{T}) where {T<:AbstractFloat}
    aval = a.val
    return result(atanh(aval), inv(one(T) - abs2(aval)), a)
end

function Base.atan(a::Measurement, b::Measurement)
    aval = a.val
    bval = b.val
    denom = abs2(aval) + abs2(bval)
    return result(atan(aval, bval),
                  (bval / denom, -aval / denom),
                  (a, b))
end

function Base.atan(a::Measurement, b::Real)
    x = a.val
    return result(atan(x, b), -b/(abs2(x) + abs2(b)), a)
end

function Base.atan(a::Real, b::Measurement)
    y = b.val
    return result(atan(a, y), -a/(abs2(a) + abs2(y)), b)
end

function Base.asec(a::Measurement)
    x = a.val
    return result(asec(x), 1 / (sqrt(x ^ 2 - 1) * abs(x)), a)
end

function Base.acsc(a::Measurement)
    x = a.val
    return result(acsc(x), -1 / (sqrt(x ^ 2 - 1) * abs(x)), a)
end

function Base.acot(a::Measurement)
    x = a.val
    return result(acot(x), -1 / (x ^ 2 + 1), a)
end

function Base.asech(a::Measurement)
    x = a.val
    return result(asech(x), -1 / (sqrt(1 - x ^ 2) * abs(x)), a)
end

function Base.acsch(a::Measurement)
    x = a.val
    return result(acsch(x), -1 / (sqrt(x ^ 2 + 1) * abs(x)), a)
end

function Base.acoth(a::Measurement)
    x = a.val
    return result(acoth(x), 1 / (1 - x ^ 2), a)
end

# Reciprocal trig functions: csc, cscd, csch, sec, secd, sech, cot, cotd, coth

function Base.csc(a::Measurement)
    aval = a.val
    val = csc(aval)
    return result(val, -val*cot(aval), a)
end

function Base.cscd(a::Measurement)
    aval = a.val
    val = cscd(aval)
    return result(val, -deg2rad(val*cotd(aval)), a)
end

function Base.csch(a::Measurement)
    aval = a.val
    val = csch(aval)
    return result(val, -val*coth(aval), a)
end

function Base.sec(a::Measurement)
    aval = a.val
    val = sec(aval)
    return result(val, val*tan(aval), a)
end

function Base.secd(a::Measurement)
    aval = a.val
    val = secd(aval)
    return result(val, deg2rad(val*tand(aval)), a)
end

function Base.sech(a::Measurement)
    aval = a.val
    val = sech(aval)
    return result(val, val*tanh(aval), a)
end

function Base.cot(a::Measurement)
    aval = a.val
    return result(cot(aval), -abs2(csc(aval)), a)
end

function Base.cotd(a::Measurement)
    aval = a.val
    return result(cotd(aval), -deg2rad(abs2(cscd(aval))), a)
end

function Base.coth(a::Measurement)
    aval = a.val
    return result(coth(aval), -abs2(csch(aval)), a)
end

### Exponential-related

# Exponentials: exp, expm1, exp10, frexp, ldexp

function Base.exp(a::Measurement)
    val = exp(a.val)
    return result(val, val, a)
end

function Base.expm1(a::Measurement)
    aval = a.val
    return result(expm1(aval), exp(aval), a)
end

function Base.exp10(a::Measurement{T}) where {T<:AbstractFloat}
    val = exp10(a.val)
    return result(val, log(T(10))*val, a)
end

function Base.frexp(a::Measurement)
    x, y = frexp(a.val)
    return (result(x, inv(exp2(y)), a), y)
end

Base.ldexp(a::Measurement{T}, e::Integer) where {T<:AbstractFloat} =
    result(ldexp(a.val, e), ldexp(one(T), e), a)

# Logarithms

function Base.log(a::Measurement, b::Measurement)
    aval = a.val
    bval = b.val
    val = log(aval, bval)
    loga = log(aval)
    return result(val, (-val / (aval * loga), 1 / (loga * bval)), (a, b))
end

function Base.log(a::Measurement) # Special case
    aval = a.val
    return result(log(aval), inv(aval), a)
end

function Base.log2(a::Measurement{T}) where {T<:AbstractFloat} # Special case
    x = a.val
    return result(log2(x), inv(log(T(2)) * x), a)
end

function Base.log10(a::Measurement{T}) where {T<:AbstractFloat} # Special case
    aval = a.val
    return result(log10(aval), inv(log(T(10)) * aval), a)
end

function Base.log1p(a::Measurement{T}) where {T<:AbstractFloat} # Special case
    aval = a.val
    return result(log1p(aval), inv(aval + one(T)), a)
end

Base.log(::Irrational{:ℯ}, a::Measurement) = log(a)

function Base.log(a::Real, b::Measurement)
    bval = b.val
    return result(log(a, bval), inv(log(a) * bval), b)
end

function Base.log(a::Measurement, b::Real)
    aval = a.val
    res = log(aval, b)
    return result(res, -res/(aval*log(aval)), a)
end

# Hypotenuse: hypot

function Base.hypot(a::Measurement, b::Measurement)
    aval = a.val
    bval = b.val
    val = hypot(aval, bval)
    return result(val,
                  (aval / val, bval / val),
                  (a, b))
end

function Base.hypot(a::Real, b::Measurement)
    bval = b.val
    res = hypot(a, bval)
    return result(res, bval / res, b)
end

function Base.hypot(a::Measurement, b::Real)
    aval = a.val
    res = hypot(aval, b)
    return result(res, aval / res, a)
end

# Square root: sqrt

function Base.sqrt(a::Measurement)
    val = sqrt(a.val)
    return result(val, inv(2 * val), a)
end

# Cube root: cbrt

function Base.cbrt(a::Measurement)
    aval = a.val
    val = cbrt(aval)
    return result(val, val / (3 * aval), a)
end

### Absolute value, sign and the likes

# Absolute value

function Base.abs(a::Measurement)
    aval = a.val
    return result(abs(aval), copysign(1, aval), a)
end

function Base.abs2(a::Measurement)
    x = a.val
    return result(abs2(x), 2*x, a)
end

# Sign: sign, copysign, flipsign

Base.sign(a::Measurement) = result(sign(a.val), 0, a)

Base.copysign(a::Measurement, b::Measurement) = ifelse(signbit(a)!=signbit(b), -a, a)
Base.copysign(a::Measurement, b::Real)        = ifelse(signbit(a)!=signbit(b), -a, a)
Base.flipsign(a::Measurement, b::Measurement) = ifelse(signbit(b), -a, a)
Base.flipsign(a::Measurement, b::Real)        = ifelse(signbit(b), -a, a)
for T in (Signed, Rational, Float32, Float64, Real)
    @eval Base.copysign(a::$T, b::Measurement) = copysign(a, b.val)
    @eval Base.flipsign(a::$T, b::Measurement) = flipsign(a, b.val)
end

### Modulo

# Use definition of "mod" function:
# http://docs.julialang.org/en/stable/manual/mathematical-operations/#division-functions
Base.mod(a::Measurement, b::Measurement) = a - fld(a, b)*b
Base.mod(a::Measurement, b::Real) = result(mod(a.val, b), 1, a)
Base.mod(a::Real, b::Measurement) = result(mod(a, b.val), -fld(a, b.val), b)

# Use definition of "rem" function:
# https://docs.julialang.org/en/latest/manual/mathematical-operations/#Division-functions-1
Base.rem(a::Measurement, b::Measurement) = a - div(a, b)*b
Base.rem(a::Measurement, b::Real) = result(rem(a.val, b), 1, a)
Base.rem(a::Real, b::Measurement) = result(rem(a, b.val), -div(a, b.val), b)
Base.rem(a::Measurement, b::Union{Measurement,Float64}, ::RoundingMode{:Nearest}) =
    a - b * round(a / b, RoundNearest)
Base.rem(a::Float64, b::Measurement, ::RoundingMode{:Nearest}) =
    a - b * round(a / b, RoundNearest)

Base.mod2pi(a::Measurement) = result(mod2pi(a.val), 1, a)
Base.rem2pi(a::Measurement, r::RoundingMode) = result(rem2pi(a.val, r), 1, a)

### Machine precision

Base.eps(::Type{Measurement{T}}) where {T<:AbstractFloat} = eps(T)
Base.eps(a::Measurement) = eps(a.val)

Base.nextfloat(a::Measurement) = result(nextfloat(a.val), 1, a)
Base.nextfloat(a::Measurement, n::Integer) = result(nextfloat(a.val, n), 1, a)

Base.maxintfloat(::Type{Measurement{T}}) where {T<:AbstractFloat} = maxintfloat(T)

Base.floatmin(::Type{Measurement{T}}) where {T<:AbstractFloat} = floatmin(T) ± zero(T)
Base.floatmax(::Type{Measurement{T}}) where {T<:AbstractFloat} = floatmax(T) ± zero(T)

Base.typemax(::Type{Measurement{T}}) where {T<:AbstractFloat} = typemax(T)

### Rounding

Base.round(a::Measurement, r::RoundingMode=RoundNearest; kwargs...) =
    measurement(round(value(a), r; kwargs...), round(uncertainty(a); kwargs...))
Base.round(::Type{T}, a::Measurement, r::RoundingMode=RoundNearest) where {T<:Integer} =
    round(T, a.val, r)
Base.floor(a::Measurement) = measurement(floor(a.val))
Base.floor(::Type{T}, a::Measurement) where {T<:Integer} = floor(T, a.val)
Base.ceil(a::Measurement) = measurement(ceil(a.val))
Base.ceil(::Type{T}, a::Measurement) where {T<:Integer} = ceil(Integer, a.val)
Base.trunc(a::Measurement) = measurement(trunc(a.val))
Base.trunc(::Type{T}, a::Measurement) where {T<:Integer} = trunc(T, a.val)

# Widening

Base.widen(::Type{Measurement{T}}) where {T<:AbstractFloat} = Measurement{widen(T)}

# To big float

Base.big(::Type{Measurement}) = Measurement{BigFloat}
Base.big(::Type{Measurement{T}}) where {T<:AbstractFloat} = Measurement{BigFloat}
Base.big(x::Measurement{<:AbstractFloat}) = convert(Measurement{BigFloat}, x)
Base.big(x::Complex{<:Measurement}) = convert(Complex{Measurement{BigFloat}}, x)

# Sum and prod

# This definition is not strictly needed, because `sum' works out-of-the-box
# with Measurement type, but this makes the function linear instead of quadratic
# in the number of arguments, but `result' is quadratic in the number of
# arguments, so in the end the function goes from cubic to quadratic.  Still not
# ideal, but this is an improvement.
Base.sum(a::AbstractArray{<:Measurement}) = result(sum(value.(a)), ones(length(a)), a)

# Same as above.  I'm not particularly proud of how the derivatives are
# computed, but something like this is needed in order to avoid errors with null
# nominal values: you may think to x ./ prod(x), but that would fail if one or
# more elements are zero.
function Base.prod(a::AbstractArray{<:Measurement})
    x = value.(a)
    return result(prod(x),
                  [prod(deleteat!(copy(x), i)) for i in eachindex(x)],
                  a)
end
