enum class MemLayout {
    ROW_MAJOR,
    COLUMN_MAJOR
}

template<
    MemLayout Layout = MemLayout::ROW_MAJOR, 
    typename T, 
    int n, 
    int... Ranks
>
class Tensor<T, n, Ranks...>: public Tensor<T, Ranks...> {
private:
    MemLayout layout;
    int dim
};
