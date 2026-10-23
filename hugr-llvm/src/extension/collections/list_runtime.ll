; Default list runtime for 64-bit targets. Lists uniquely own their byte buffer.
; Elements are moved/copied through memcpy, so their buffer alignment is irrelevant.
; get copies only copyable elements; discard_empty checks emptiness before freeing.
%hugr.list = type { ptr, i64, i64, i64 }
declare ptr @__hugr_list_allocate(i64)
declare void @__hugr_list_free(ptr)
declare void @__hugr_list_allocation_failure()
declare void @__hugr_list_capacity_failure()
declare void @__hugr_list_nonempty_failure()
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1 immarg)
declare void @llvm.memmove.p0.p0.i64(ptr, ptr, i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)

; Fail before allocating or indexing if capacity * element size overflows.
define internal ptr @hugr_list_alloc(i64 %capacity, i64 %size) {
entry:
  %product = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %capacity, i64 %size)
  %bytes = extractvalue { i64, i1 } %product, 0
  %overflow = extractvalue { i64, i1 } %product, 1
  br i1 %overflow, label %overflow_fail, label %alloc
alloc:
  %empty = icmp eq i64 %bytes, 0
  %nonzero = select i1 %empty, i64 1, i64 %bytes
  %data = call ptr @__hugr_list_allocate(i64 %nonzero)
  %null = icmp eq ptr %data, null
  br i1 %null, label %fail, label %done
fail:
  call void @__hugr_list_allocation_failure()
  unreachable
overflow_fail:
  call void @__hugr_list_capacity_failure()
  unreachable
done:
  ret ptr %data
}

define ptr @__rt__list__new(i64 %capacity, i64 %size, i64 %alignment, ptr %destructor) {
entry:
  %data = call ptr @hugr_list_alloc(i64 %capacity, i64 %size)
  %end = getelementptr %hugr.list, ptr null, i64 1
  %header_size = ptrtoint ptr %end to i64
  %list = call ptr @__hugr_list_allocate(i64 %header_size)
  %null = icmp eq ptr %list, null
  br i1 %null, label %fail, label %init
fail:
  call void @__hugr_list_free(ptr %data)
  call void @__hugr_list_allocation_failure()
  unreachable
init:
  %dp = getelementptr %hugr.list, ptr %list, i32 0, i32 0
  %lp = getelementptr %hugr.list, ptr %list, i32 0, i32 1
  %cp = getelementptr %hugr.list, ptr %list, i32 0, i32 2
  %sp = getelementptr %hugr.list, ptr %list, i32 0, i32 3
  store ptr %data, ptr %dp
  store i64 0, ptr %lp
  store i64 %capacity, ptr %cp
  store i64 %size, ptr %sp
  ret ptr %list
}

; Return a slot in a buffer large enough for one further element.
define internal ptr @hugr_list_grow(ptr %list) {
entry:
  %dp = getelementptr %hugr.list, ptr %list, i32 0, i32 0
  %lp = getelementptr %hugr.list, ptr %list, i32 0, i32 1
  %cp = getelementptr %hugr.list, ptr %list, i32 0, i32 2
  %sp = getelementptr %hugr.list, ptr %list, i32 0, i32 3
  %len = load i64, ptr %lp
  %cap = load i64, ptr %cp
  %size = load i64, ptr %sp
  %data = load ptr, ptr %dp
  %full = icmp eq i64 %len, %cap
  br i1 %full, label %grow, label %done
grow:
  %too_big = icmp uge i64 %cap, 9223372036854775808
  br i1 %too_big, label %fail, label %resize
fail:
  call void @__hugr_list_capacity_failure()
  unreachable
resize:
  %double = mul i64 %cap, 2
  %empty = icmp eq i64 %cap, 0
  %next = select i1 %empty, i64 1, i64 %double
  %new_data = call ptr @hugr_list_alloc(i64 %next, i64 %size)
  %live_bytes = mul i64 %len, %size
  call void @llvm.memcpy.p0.p0.i64(ptr %new_data, ptr %data, i64 %live_bytes, i1 false)
  call void @__hugr_list_free(ptr %data)
  store ptr %new_data, ptr %dp
  store i64 %next, ptr %cp
  br label %done
done:
  %buffer = phi ptr [ %data, %entry ], [ %new_data, %resize ]
  ret ptr %buffer
}

define void @__rt__list__push(ptr %list, ptr %elem) {
entry:
  %lp = getelementptr %hugr.list, ptr %list, i32 0, i32 1
  %sp = getelementptr %hugr.list, ptr %list, i32 0, i32 3
  %len = load i64, ptr %lp
  %size = load i64, ptr %sp
  %data = call ptr @hugr_list_grow(ptr %list)
  %offset = mul i64 %len, %size
  %slot = getelementptr i8, ptr %data, i64 %offset
  call void @llvm.memcpy.p0.p0.i64(ptr %slot, ptr %elem, i64 %size, i1 false)
  %next = add i64 %len, 1
  store i64 %next, ptr %lp
  ret void
}

define i64 @__rt__list__length(ptr %list) {
entry:
  %lp = getelementptr %hugr.list, ptr %list, i32 0, i32 1
  %len = load i64, ptr %lp
  ret i64 %len
}

define i1 @__rt__list__get(ptr %list, i64 %idx, ptr %out) {
entry:
  %len = call i64 @__rt__list__length(ptr %list)
  %ok = icmp ult i64 %idx, %len
  br i1 %ok, label %copy, label %done
copy:
  %dp = getelementptr %hugr.list, ptr %list, i32 0, i32 0
  %sp = getelementptr %hugr.list, ptr %list, i32 0, i32 3
  %data = load ptr, ptr %dp
  %size = load i64, ptr %sp
  %offset = mul i64 %idx, %size
  %slot = getelementptr i8, ptr %data, i64 %offset
  call void @llvm.memcpy.p0.p0.i64(ptr %out, ptr %slot, i64 %size, i1 false)
  br label %done
done:
  ret i1 %ok
}

define i1 @__rt__list__pop(ptr %list, ptr %out) {
entry:
  %len = call i64 @__rt__list__length(ptr %list)
  %empty = icmp eq i64 %len, 0
  br i1 %empty, label %none, label %some
none:
  ret i1 false
some:
  %idx = sub i64 %len, 1
  %ok = call i1 @__rt__list__get(ptr %list, i64 %idx, ptr %out)
  %lp = getelementptr %hugr.list, ptr %list, i32 0, i32 1
  store i64 %idx, ptr %lp
  ret i1 true
}

; Swap the owned input element with the old element, preserving the external ABI.
define i1 @__rt__list__set(ptr %list, i64 %idx, ptr %elem) {
entry:
  %len = call i64 @__rt__list__length(ptr %list)
  %ok = icmp ult i64 %idx, %len
  br i1 %ok, label %replace, label %done
replace:
  %dp = getelementptr %hugr.list, ptr %list, i32 0, i32 0
  %sp = getelementptr %hugr.list, ptr %list, i32 0, i32 3
  %data = load ptr, ptr %dp
  %size = load i64, ptr %sp
  %offset = mul i64 %idx, %size
  %slot = getelementptr i8, ptr %data, i64 %offset
  br label %loop
loop:
  %i = phi i64 [ 0, %replace ], [ %next, %swap ]
  %more = icmp ult i64 %i, %size
  br i1 %more, label %swap, label %done
swap:
  %oldp = getelementptr i8, ptr %slot, i64 %i
  %newp = getelementptr i8, ptr %elem, i64 %i
  %old = load i8, ptr %oldp
  %new = load i8, ptr %newp
  store i8 %new, ptr %oldp
  store i8 %old, ptr %newp
  %next = add i64 %i, 1
  br label %loop
done:
  ret i1 %ok
}

define i1 @__rt__list__insert(ptr %list, i64 %idx, ptr %elem) {
entry:
  %len = call i64 @__rt__list__length(ptr %list)
  %ok = icmp ule i64 %idx, %len
  br i1 %ok, label %insert, label %done
insert:
  %sp = getelementptr %hugr.list, ptr %list, i32 0, i32 3
  %size = load i64, ptr %sp
  %data = call ptr @hugr_list_grow(ptr %list)
  %offset = mul i64 %idx, %size
  %slot = getelementptr i8, ptr %data, i64 %offset
  %next_slot = getelementptr i8, ptr %slot, i64 %size
  %tail_len = sub i64 %len, %idx
  %tail_bytes = mul i64 %tail_len, %size
  call void @llvm.memmove.p0.p0.i64(ptr %next_slot, ptr %slot, i64 %tail_bytes, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %slot, ptr %elem, i64 %size, i1 false)
  %lp = getelementptr %hugr.list, ptr %list, i32 0, i32 1
  %next = add i64 %len, 1
  store i64 %next, ptr %lp
  br label %done
done:
  ret i1 %ok
}

define void @__rt__list__discard_empty(ptr %list) {
entry:
  %len = call i64 @__rt__list__length(ptr %list)
  %empty = icmp eq i64 %len, 0
  br i1 %empty, label %dispose, label %fail
fail:
  call void @__hugr_list_nonempty_failure()
  unreachable
dispose:
  %dp = getelementptr %hugr.list, ptr %list, i32 0, i32 0
  %data = load ptr, ptr %dp
  call void @__hugr_list_free(ptr %data)
  call void @__hugr_list_free(ptr %list)
  ret void
}
