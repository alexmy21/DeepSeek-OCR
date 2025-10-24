"""
MIT License

Copyright (c) 2023: Jakob Nybo Nissen.

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

https://github.com/jakobnissen/Probably.jl/blob/master/src/hyperloglog/hyperloglog.jl

"""

"""
HyperLogLog Set Implementation with Extended Set Operations

This module provides a HyperLogLog (HLL) implementation with additional set operations
including union, intersection, difference, and complement operations.

Key Features:
- Efficient cardinality estimation using HLL algorithm
- Set operations between HLL sets
- Serialization/deserialization support
- Similarity measures (Jaccard, Cosine)
"""

module HllSets

    # NOTE: The user must have a `constants.jl` file with RAW_ARRAYS and BIAS_ARRAYS
    # and the JSON3.jl and SHA.jl packages installed.
    include("constants.jl")
    using SHA
    using JSON3

    export HllSet, add!, count, union, intersect, diff, isequal, isempty, id, delta, getbin, getzeros, maxidx, match, cosine, dump, restore, to_binary_tensor, flatten_tensor, tensor_to_string, string_to_tensor, binary_tensor_to_hllset, set_xor, set_comp, set_added, set_deleted

    struct HllSet{P}
        counts::Vector{UInt32}

        function HllSet{P}() where {P}
            isa(P, Integer) || throw(ArgumentError("P must be integer"))
            (P < 4 || P > 18) && throw(ArgumentError("P must be between 4 and 18"))
            new(fill(UInt32(0), 2^P))
        end
    end

    function HllSet(p::Int=10)
        return HllSet{p}()
    end

    # Core HLL Operations --------------------------------------------------------

    """
        add!(hll::HllSet{P}, x::Any; seed::UInt64=0)

    Add an element to the HLL set.
    """
    function add!(hll::HllSet{P}, x::Any; seed::UInt64 = 0x0) where {P}
        h = u_hash(x; seed=seed)
        bin = getbin(h; P=P)
        # The number of leading zeros + 1 is the rank to be stored
        idx = getzeros(h; P=P)
        if idx <= 32
            hll.counts[bin] |= (UInt32(1) << (idx - 1))
        end
    end

    function add!(hll::HllSet{P}, values::Union{Set, Vector}; seed::UInt64 = 0x0) where {P}
        for value in values
            add!(hll, value, seed=seed)
        end
    end    

    # Helper Functions ----------------------------------------------------------

    function _validate_compatible(x::HllSet{P}, y::HllSet{P}) where {P}
        length(x.counts) == length(y.counts) || 
            throw(ArgumentError("HLL sets must have same precision P"))
    end
    
    ### IMPROVED: Added a clear helper function
    """
        num_registers(hll::HllSet)

    Returns the number of registers (bins) in the HllSet, which is 2^P.
    """
    num_registers(::HllSet{P}) where {P} = 1 << P
    num_registers(::Type{HllSet{P}}) where {P} = 1 << P

    # Set Operations ------------------------------------------------------------

    """
        union(x::HllSet{P}, y::HllSet{P}) where {P}

    Compute union of two HLL sets.
    """
    function Base.union!(dest::HllSet{P}, src::HllSet{P}) where {P}
        _validate_compatible(dest, src)
        @inbounds for i in eachindex(dest.counts)
            ### IMPROVED: Removed redundant broadcasting dot
            dest.counts[i] |= src.counts[i]
        end
        return dest
    end

    function Base.union(x::HllSet{P}, y::HllSet{P}) where {P} 
        _validate_compatible(x, y)
        z = HllSet{P}()
        @inbounds for i in eachindex(x.counts)
            ### IMPROVED: Removed redundant broadcasting dot
            z.counts[i] = x.counts[i] | y.counts[i]
        end
        return z
    end

    """
        intersect(x::HllSet{P}, y::HllSet{P}) where {P}

    Compute intersection of two HLL sets.
    """
    function Base.intersect(x::HllSet{P}, y::HllSet{P}) where {P} 
        _validate_compatible(x, y)
        z = HllSet{P}()
        @inbounds for i in eachindex(x.counts)
            z.counts[i] = x.counts[i] & y.counts[i]
        end
        return z
    end

    """
        diff(hll_1::HllSet{P}, hll_2::HllSet{P}) where {P}

    Compute the set difference between two HLL sets.
    Returns a named tuple with:
    - `left_exclusive`: Elements in hll_1 but not in hll_2.
    - `intersection`: Elements in both hll_1 and hll_2.
    - `right_exclusive`: Elements in hll_2 but not in hll_1.
    """
    function Base.diff(hll_1::HllSet{P}, hll_2::HllSet{P}) where {P}
        _validate_compatible(hll_1, hll_2)
        
        # Elements in hll_1 but not in hll_2
        left_exclusive = set_comp(hll_1, hll_2)
        # Elements in hll_2 but not in hll_1
        right_exclusive = set_comp(hll_2, hll_1)
        # Elements in both
        intersection = intersect(hll_1, hll_2)

        return (left_exclusive = left_exclusive, intersection = intersection, right_exclusive = right_exclusive)
    end

    """
        set_comp(x::HllSet{P}, y::HllSet{P}) where {P}

    Compute the elements of `x` that are not in `y`. (i.e., `x \\ y`).
    """
    function set_comp(x::HllSet{P}, y::HllSet{P}) where {P} 
        _validate_compatible(x, y)
        z = HllSet{P}()
        @inbounds for i in eachindex(x.counts)
            ### IMPROVED: Simplified bitwise logic
            z.counts[i] = x.counts[i] & ~y.counts[i]
        end
        return z
    end

    """
        set_xor(x::HllSet{P}, y::HllSet{P}) where {P}

    Compute the symmetric difference between two HLL sets.
    """
    function set_xor(x::HllSet{P}, y::HllSet{P}) where {P} 
        _validate_compatible(x, y)
        z = HllSet{P}()
        @inbounds for i in eachindex(x.counts)
            ### IMPROVED: Removed redundant broadcasting dot
            z.counts[i] = xor(x.counts[i], y.counts[i])
        end
        return z
    end

    ### IMPROVED: Replaced copy!(src) with standard copy(src)
    """
        copy(src::HllSet{P})

    Create a new, independent copy of an HllSet.
    """
    function Base.copy(src::HllSet{P}) where {P}
        dest = HllSet{P}()
        copyto!(dest.counts, src.counts)
        return dest
    end

    """
        copy!(dest::HllSet{P}, src::HllSet{P})

    Copy the contents of `src` into `dest`. `dest` must be a pre-existing HllSet.
    """
    function Base.copy!(dest::HllSet{P}, src::HllSet{P}) where {P}
        _validate_compatible(dest, src)
        copyto!(dest.counts, src.counts)
        return dest
    end

    """
        isequal(x::HllSet{P}, y::HllSet{P}) where {P}

    Check if two HLL sets are equal (i.e., have identical registers).
    """    
    function Base.isequal(x::HllSet{P}, y::HllSet{P}) where {P} 
        _validate_compatible(x, y)
        return x.counts == y.counts
    end    

    Base.isempty(x::HllSet{P}) where {P} = all(iszero, x.counts)   

    """
        count(x::HllSet{P}) where {P}

    Estimate the cardinality of the HLL set.
    """
    ### IMPROVED: Fixed critical bug in cardinality calculation
    function Base.count(x::HllSet{P}) where {P}
        m = num_registers(x) # Correctly get number of registers
        harmonic_sum = sum(inv(1 << maxidx(i)) for i in x.counts)
        harmonic_mean = m / harmonic_sum
        
        biased_estimate = α(x) * m * harmonic_mean
        return round(Int, biased_estimate - bias(x, biased_estimate))
    end

    """
        Set of helper functions for cardinality estimation.
    """
    ### IMPROVED: Now uses UInt64 and is much faster
    """
        getbin(x::UInt64; P::Int)

    Extracts the top `P` bits from a 64-bit hash `x` to determine the register index.
    """
    function getbin(x::UInt64; P::Int=10)
        return Int((x >>> (64 - P)) + 1) # +1 for 1-based indexing
    end

    ### IMPROVED: Now uses UInt64
    """
        getzeros(x::UInt64; P::Int)

    Counts the number of leading zeros in the lower `64-P` bits of the hash `x`.
    """
    function getzeros(x::UInt64; P::Int=10)
        # Mask to get the lower bits used for zero-counting
        lower_bits_mask = (UInt64(1) << (64 - P)) - 1
        # Count trailing zeros of the lower bits. If all are zero, count is 64-P.
        # We add 1 because HLL rank is 1-based.
        return trailing_zeros(x & lower_bits_mask) + 1
    end

    ### IMPROVED: Simplified logic
    α(x::HllSet{P}) where {P} = if P == 4; 0.673
        elseif P == 5; 0.697
        elseif P == 6; 0.709
        else; 0.7213 / (1 + 1.079 / num_registers(x))
        end 
    
    function bias(::HllSet{P}, biased_estimate) where {P}
        if P < 4 || P > 18
            error("We only have bias estimates for P ∈ 4:18")
        end
        rawarray = @inbounds RAW_ARRAYS[P - 3]
        biasarray = @inbounds BIAS_ARRAYS[P - 3]
        firstindex = searchsortedfirst(rawarray, biased_estimate)
        if firstindex == length(rawarray) + 1
            return 0.0
        elseif firstindex == 1
            return @inbounds biasarray[1]
        else
            x1, x2 = @inbounds rawarray[firstindex - 1], @inbounds rawarray[firstindex]
            y1, y2 = @inbounds biasarray[firstindex - 1], @inbounds biasarray[firstindex]
            delta = @fastmath (biased_estimate - x1) / (x2 - x1)
            return y1 + delta * (y2 - y1)
        end
    end

    ### IMPROVED: Simplified implementation
    """
        maxidx(x::UInt32)

    Returns the position of the most significant bit (MSB) in a 32-bit register.
    This corresponds to the maximum rank stored in that register.
    """
    function maxidx(x::UInt32)        
        # If x is 0, there are no bits set, so the rank is 0.
        # Otherwise, find the position of the MSB.
        return x == 0 ? 0 : (sizeof(UInt32) * 8 - leading_zeros(x))
    end

    # Match Operations ------------------------------------------------------------

    """
        match(x::HllSet{P}, y::HllSet{P}) where {P}
    Compute the Jaccard similarity between two HLL sets as a percentage (0-100).
    """
    function Base.match(x::HllSet{P}, y::HllSet{P}) where {P}
        _validate_compatible(x, y)
        
        count_u = count(union(x, y))
        # Avoid division by zero if both sets are empty
        count_u == 0 && return 100

        count_i = count(intersect(x, y))
        return round(Int64, (count_i / count_u) * 100)
    end

    """
        cosine(hll_1::HllSet{P}, hll_2::HllSet{P}) where {P}
    Compute the cosine similarity between two HLL sets.
    """
    function cosine(hll_1::HllSet{P}, hll_2::HllSet{P}) where {P}
        _validate_compatible(hll_1, hll_2)

        v1 = hll_1.counts
        v2 = hll_2.counts
        norm_v1 = norm(v1)
        norm_v2 = norm(v2)
        
        if norm_v1 == 0 || norm_v2 == 0
            return 0.0
        end
        return dot(v1, v2) / (norm_v1 * norm_v2)
    end

    # Serialization and Deserialization ----------------------------------------

    """
        to_binary_tensor(hll::HllSet{P}) where {P}
    Convert the HLL set to a binary tensor (Matrix of Bools).
    """
    function to_binary_tensor(hll::HllSet{P}) where {P}
        m = num_registers(hll)
        tensor = falses(m, 32)
        for i in 1:m
            # reinterpret is a fast way to get bits
            bits = reinterpret(Bool, hll.counts[i])
            # Note: bitstring is little-endian, so we reverse to match intuition
            tensor[i, :] .= reverse(bits)
        end
        return tensor
    end

    """
        flatten_tensor(tensor::Array{Bool, 2})
    Flatten the binary tensor to a 1D array.
    """
    flatten_tensor(tensor::Array{Bool, 2}) = vec(tensor)

    """
        tensor_to_string(flattened_tensor::Vector{Bool})
    Convert a flattened binary tensor to a string representation.
    """
    tensor_to_string(flattened_tensor::Vector{Bool}) = join([b ? "1" : "0" for b in flattened_tensor], "")

    """
        string_to_tensor(str::String, P::Int=10)
    Convert a string representation of a binary tensor to a 2D array.
    """
    function string_to_tensor(str::String, P::Int=10)
        m = 2^P
        expected_len = m * 32
        if length(str) != expected_len
            str = rpad(str, expected_len, "0")
        end
        return reshape([c == '1' for c in str], (m, 32))
    end

    """
        binary_tensor_to_hllset(tensor::Array{Bool, 2}, P::Int=10)
    Convert a binary tensor to an HLL set.
    """
    function binary_tensor_to_hllset(tensor::Array{Bool, 2}, P::Int=10)
        hll = HllSet{P}()
        for i in 1:num_registers(hll)
            # Reverse bits back to little-endian for reinterpret
            bits_str = join(reverse(tensor[i, :]))
            hll.counts[i] = parse(UInt32, bits_str, base=2)
        end
        return hll
    end   
   
    # HllSet ID ---------------------------------------------------
    
    """
        id(x::HllSet{P}) where {P}
    Compute a stable SHA1 hash of the HllSet's contents, useful for identification.
    """
    function id(x::HllSet{P}) where {P}
        isnothing(x) && return nothing
        bytearray = reinterpret(UInt8, x.counts)
        hash_value = SHA.sha1(bytearray)
        return SHA.bytes2hex(hash_value)
    end

    ### IMPROVED: Now a safe, efficient alias for id()
    function sha1(x::HllSet{P}) where {P}
        @warn "sha1(hll) is deprecated in favor of id(hll) for clarity." maxlog=1
        return id(x)
    end

    ### IMPROVED: Now uses UInt64 and warns about stability
    """
        u_hash(x; seed::UInt64 = 0x0)

    Computes a 64-bit unsigned hash for an element `x`.
    **WARNING:** Uses Julia's built-in `hash`, which is not guaranteed to be stable
    across different Julia versions. For applications requiring serialization and
    deserialization, a stable hash like SHA-256 should be used instead.
    """
    function u_hash(x; seed::UInt64 = 0x0) 
        h = reinterpret(UInt64, hash(x)) ⊻ seed
        return h
    end

    # Overload the show function to print the HllSet --------------------------------------------------
    Base.show(io::IO, x::HllSet{P}) where {P} = print(io, "HllSet{$(P)}($(count(x)) estimated elements)")

    ### IMPROVED: Removed confusing sizeof overloads
    # Base.sizeof(::Type{HllSet{P}}) where {P} = 1 << P
    # Base.sizeof(x::HllSet{P}) where {P} = sizeof(typeof(x))

    # Depricated Functions --------------------------------------------------
    
    function Base.dump(x::HllSet{P}) where {P}
        @warn "dump(hll) is deprecated, access hll.counts directly." maxlog=1
        return x.counts
    end

    ### IMPROVED: Fixed bugs and used fast copyto!
    function restore!(z::HllSet{P}, x::Vector{UInt32}) where {P} 
        if length(x) != num_registers(z)
            error("The length of the vector must be equal to the number of HllSet registers.")
        end        
        copyto!(z.counts, x)
        return z
    end

    function restore!(z::HllSet{P}, x::String) where {P}
        dataset = JSON3.read(x, Vector{UInt32})
        if length(dataset) != num_registers(z)
            error("JSON data length does not match HllSet size.")
        end
        copyto!(z.counts, dataset)
        return z
    end 

end