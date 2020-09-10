#ifndef COMBINATORIAL_FACE_STRUCTURE
#define COMBINATORIAL_FACE_STRUCTURE

#include "bitsets.h"
/*
This structure is subject to change.
Use the access functions instead.
*/
struct face_struct{
    // All atoms that are contained in the face.
    uint64_t* atom_rep;
    // Coatoms such that their meet produces the face.
    // Note that in a simple polytope this is unique except for the empty face.
    uint64_t* coatom_gen;

    size_t face_length;
    size_t face_length_coatom_gen;
    int coatom_gen_is_maximal;
};

/*
#############################################################################
# Underlying bitset functions access
#############################################################################
*/

inline int face_cmp(face_struct& a, face_struct& b){
    return bitset_cmp(a.atom_rep, b.atom_rep, a.face_length);
}

inline void face_add_atom(face_struct& face, size_t n){
    bitset_add(face.atom_rep, n);
}

inline void face_discard_atom(face_struct& face, size_t n){
    bitset_discard(face.atom_rep, n);
}

inline void face_add_coatom(face_struct& face, size_t n){
    bitset_add(face.coatom_gen, n);
}

inline void face_set_first_n_atoms(face_struct& face, size_t n){
    bitset_set_first_n(face.atom_rep, face.face_length, n);
}

inline int atom_in_face(face_struct& face, int n){
    return bitset_in(face.atom_rep, n);
}

inline int is_zero(face_struct& face){
    return bitset_isempty(face.atom_rep, face.face_length);
}

inline size_t face_next_atom(face_struct& face, size_t n){
    return bitset_next(face.atom_rep, face.face_length, n);
}

inline size_t count_atoms(face_struct& face){
    return count_atoms(face.atom_rep, face.face_length);
}

inline int test_alignment(face_struct& face){
    return (test_alignment(face.atom_rep) && test_alignment(face.coatom_gen));
}

/*
#############################################################################
# Face Initalization
#############################################################################
*/

void face_clear(face_struct& face);

void face_copy(face_struct& dst, face_struct& src);

int initalize_face_with_allocate_instructions(\
        face_struct& face, size_t n_faces, size_t n_atoms, size_t step,
        void* location, size_t* alignment, size_t* size);
    /*
    Initialize a face step by step.
    Return ``1`` as long as this function requests an aligned malloc.
    */

/*
#############################################################################
# Basic Access
#############################################################################
*/

int face_add_atom_safe(face_struct& face, size_t n);

inline void set_coatom_gen_maximal(face_struct& face, int val){
    face.coatom_gen_is_maximal = val;
}

/*
#############################################################################
# Arithmetic
#############################################################################
*/

inline int is_subset(face_struct& A, face_struct& B){
    if (A.coatom_gen_is_maximal && B.coatom_gen_is_maximal){
        return is_subset(B.coatom_gen, A.coatom_gen, A.face_length_coatom_gen);
    }
    return is_subset(A.atom_rep, B.atom_rep, A.face_length);
}

inline void intersection(face_struct& dest, face_struct& A, face_struct& B){
    intersection(dest.atom_rep, A.atom_rep, B.atom_rep, dest.face_length);
    unite(dest.coatom_gen, A.coatom_gen, B.coatom_gen, dest.face_length_coatom_gen);
    dest.coatom_gen_is_maximal = 0;
}

#endif