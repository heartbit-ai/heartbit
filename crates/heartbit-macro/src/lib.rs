use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    FnArg, ItemFn, LitStr, Meta, Pat, ReturnType, Type, parse_macro_input, punctuated::Punctuated,
    token::Comma,
};

/// Derive a `Tool` implementation from an async function.
///
/// # Attribute syntax
///
/// ```ignore
/// #[heartbit_tool(description = "Search for files matching a pattern")]
/// async fn search_files(
///     /// The glob pattern to match
///     pattern: String,
///     /// Maximum results to return
///     #[tool(default = 10)]
///     max_results: Option<u32>,
/// ) -> Result<ToolOutput, Error> {
///     // ...
/// }
/// ```
///
/// Generates a unit struct `SearchFiles` that implements `heartbit::Tool`.
#[proc_macro_attribute]
pub fn heartbit_tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr_args = parse_macro_input!(attr as ToolAttr);
    let func = parse_macro_input!(item as ItemFn);

    match expand_heartbit_tool(attr_args, func) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

struct ToolAttr {
    description: String,
}

impl syn::parse::Parse for ToolAttr {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut description = None;

        while !input.is_empty() {
            let ident: syn::Ident = input.parse()?;
            if ident == "description" {
                let _eq: syn::Token![=] = input.parse()?;
                let lit: LitStr = input.parse()?;
                description = Some(lit.value());
            } else {
                return Err(syn::Error::new(ident.span(), "unknown attribute"));
            }

            if !input.is_empty() {
                let _comma: syn::Token![,] = input.parse()?;
            }
        }

        let description = description.ok_or_else(|| {
            syn::Error::new(proc_macro2::Span::call_site(), "missing `description`")
        })?;

        Ok(ToolAttr { description })
    }
}

struct ParamInfo {
    name: syn::Ident,
    ty: Type,
    doc: Option<String>,
    is_option: bool,
    default_value: Option<syn::Lit>,
}

fn expand_heartbit_tool(attr: ToolAttr, func: ItemFn) -> syn::Result<TokenStream2> {
    if func.sig.asyncness.is_none() {
        return Err(syn::Error::new_spanned(
            func.sig.fn_token,
            "function must be async",
        ));
    }

    let fn_name = &func.sig.ident;
    let tool_name_str = fn_name.to_string();

    // Convert snake_case fn name to PascalCase struct name
    let struct_name = format_ident!(
        "{}",
        tool_name_str
            .split('_')
            .map(|s| {
                let mut c = s.chars();
                match c.next() {
                    Some(first) => {
                        let upper: String = first.to_uppercase().collect();
                        format!("{upper}{}", c.as_str())
                    }
                    None => String::new(),
                }
            })
            .collect::<String>()
    );

    let params = extract_params(&func.sig.inputs)?;
    let schema = build_schema_tokens(&params)?;
    let deserialize_tokens = build_deserialize_tokens(&params);
    let call_args: Vec<_> = params.iter().map(|p| &p.name).collect();
    let description = &attr.description;

    // Check return type — must return something (we don't enforce exact type)
    if matches!(func.sig.output, ReturnType::Default) {
        return Err(syn::Error::new_spanned(
            &func.sig,
            "function must have a return type (e.g., Result<ToolOutput, Error>)",
        ));
    }

    // Strip #[tool(...)] and doc attrs from params before emitting the function
    let mut clean_func = func.clone();
    for arg in &mut clean_func.sig.inputs {
        if let FnArg::Typed(pat_type) = arg {
            pat_type
                .attrs
                .retain(|a| !a.path().is_ident("tool") && !a.path().is_ident("doc"));
        }
    }

    Ok(quote! {
        #clean_func

        pub struct #struct_name;

        impl ::heartbit::Tool for #struct_name {
            fn definition(&self) -> ::heartbit::ToolDefinition {
                ::heartbit::ToolDefinition {
                    name: #tool_name_str.to_string(),
                    description: #description.to_string(),
                    input_schema: #schema,
                }
            }

            fn execute(
                &self,
                _ctx: &::heartbit::ExecutionContext,
                input: ::serde_json::Value,
            ) -> ::std::pin::Pin<
                ::std::boxed::Box<
                    dyn ::std::future::Future<
                        Output = ::std::result::Result<::heartbit::ToolOutput, ::heartbit::Error>,
                    > + Send + '_,
                >,
            > {
                Box::pin(async move {
                    #deserialize_tokens
                    #fn_name(#(#call_args),*).await
                })
            }
        }
    })
}

fn extract_params(inputs: &Punctuated<FnArg, Comma>) -> syn::Result<Vec<ParamInfo>> {
    let mut params = Vec::new();

    for arg in inputs {
        let FnArg::Typed(pat_type) = arg else {
            return Err(syn::Error::new_spanned(
                arg,
                "self parameters are not supported",
            ));
        };

        let Pat::Ident(pat_ident) = pat_type.pat.as_ref() else {
            return Err(syn::Error::new_spanned(
                &pat_type.pat,
                "only simple ident patterns supported",
            ));
        };

        let name = pat_ident.ident.clone();
        let ty = (*pat_type.ty).clone();
        let is_option = is_option_type(&ty);

        // Extract doc comments from attrs
        let doc = pat_type
            .attrs
            .iter()
            .filter_map(|a| {
                if a.path().is_ident("doc")
                    && let Meta::NameValue(nv) = &a.meta
                    && let syn::Expr::Lit(expr_lit) = &nv.value
                    && let syn::Lit::Str(s) = &expr_lit.lit
                {
                    Some(s.value().trim().to_string())
                } else {
                    None
                }
            })
            .reduce(|a, b| format!("{a} {b}"));

        // Extract #[tool(default = ...)] from attrs
        let default_value = pat_type
            .attrs
            .iter()
            .filter(|a| a.path().is_ident("tool"))
            .find_map(|a| {
                a.parse_args_with(|input: syn::parse::ParseStream| {
                    let ident: syn::Ident = input.parse()?;
                    if ident != "default" {
                        return Err(syn::Error::new(ident.span(), "expected `default`"));
                    }
                    let _eq: syn::Token![=] = input.parse()?;
                    let lit: syn::Lit = input.parse()?;
                    Ok(lit)
                })
                .ok()
            });

        params.push(ParamInfo {
            name,
            ty,
            doc,
            is_option,
            default_value,
        });
    }

    Ok(params)
}

fn is_option_type(ty: &Type) -> bool {
    if let Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last()
    {
        return seg.ident == "Option";
    }
    false
}

fn type_to_schema_tokens(ty: &Type) -> TokenStream2 {
    if let Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last()
    {
        let ident_str = seg.ident.to_string();

        // Handle Option<T> — extract inner type
        if ident_str == "Option" {
            if let syn::PathArguments::AngleBracketed(args) = &seg.arguments
                && let Some(syn::GenericArgument::Type(inner)) = args.args.first()
            {
                return type_to_schema_tokens(inner);
            }
            return quote! { ::serde_json::json!({}) };
        }

        // Handle Vec<T>
        if ident_str == "Vec" {
            if let syn::PathArguments::AngleBracketed(args) = &seg.arguments
                && let Some(syn::GenericArgument::Type(inner)) = args.args.first()
            {
                let items = type_to_schema_tokens(inner);
                return quote! { ::serde_json::json!({"type": "array", "items": #items}) };
            }
            return quote! { ::serde_json::json!({"type": "array"}) };
        }

        // Handle Value (serde_json::Value) — any
        if ident_str == "Value" {
            return quote! { ::serde_json::json!({}) };
        }

        match ident_str.as_str() {
            "String" | "str" => return quote! { ::serde_json::json!({"type": "string"}) },
            "i8" | "i16" | "i32" | "i64" | "i128" | "u8" | "u16" | "u32" | "u64" | "u128"
            | "usize" | "isize" => return quote! { ::serde_json::json!({"type": "integer"}) },
            "f32" | "f64" => return quote! { ::serde_json::json!({"type": "number"}) },
            "bool" => return quote! { ::serde_json::json!({"type": "boolean"}) },
            _ => {}
        }
    }

    // Fallback: any
    quote! { ::serde_json::json!({}) }
}

fn build_schema_tokens(params: &[ParamInfo]) -> syn::Result<TokenStream2> {
    let mut property_entries = Vec::new();
    let mut required_names = Vec::new();

    for param in params {
        let name_str = param.name.to_string();
        let type_schema = type_to_schema_tokens(&param.ty);

        let prop = if let Some(doc) = &param.doc {
            if let Some(default_val) = &param.default_value {
                let default_token = lit_to_json_token(default_val);
                quote! {
                    {
                        let mut s = #type_schema;
                        if let Some(obj) = s.as_object_mut() {
                            obj.insert("description".to_string(), ::serde_json::json!(#doc));
                            obj.insert("default".to_string(), ::serde_json::json!(#default_token));
                        }
                        props.insert(#name_str.to_string(), s);
                    }
                }
            } else {
                quote! {
                    {
                        let mut s = #type_schema;
                        if let Some(obj) = s.as_object_mut() {
                            obj.insert("description".to_string(), ::serde_json::json!(#doc));
                        }
                        props.insert(#name_str.to_string(), s);
                    }
                }
            }
        } else if let Some(default_val) = &param.default_value {
            let default_token = lit_to_json_token(default_val);
            quote! {
                {
                    let mut s = #type_schema;
                    if let Some(obj) = s.as_object_mut() {
                        obj.insert("default".to_string(), ::serde_json::json!(#default_token));
                    }
                    props.insert(#name_str.to_string(), s);
                }
            }
        } else {
            quote! {
                props.insert(#name_str.to_string(), #type_schema);
            }
        };

        property_entries.push(prop);

        if !param.is_option {
            required_names.push(name_str);
        }
    }

    Ok(quote! {
        {
            let mut props = ::serde_json::Map::new();
            #(#property_entries)*
            let mut schema = ::serde_json::json!({
                "type": "object",
                "properties": ::serde_json::Value::Object(props),
            });
            let required = ::serde_json::json!([#(#required_names),*]);
            if let Some(arr) = required.as_array() {
                if !arr.is_empty() {
                    schema.as_object_mut().unwrap().insert("required".to_string(), required);
                }
            }
            schema
        }
    })
}

fn build_deserialize_tokens(params: &[ParamInfo]) -> TokenStream2 {
    let extractions: Vec<_> = params
        .iter()
        .map(|p| {
            let name = &p.name;
            let name_str = name.to_string();
            let ty = &p.ty;

            if p.is_option {
                if let Some(default_val) = &p.default_value {
                    let default_json = lit_to_json_token(default_val);
                    quote! {
                        let #name: #ty = match input.get(#name_str) {
                            Some(v) if !v.is_null() => Some(
                                ::serde_json::from_value(v.clone())
                                    .map_err(|e| ::heartbit::Error::Agent(
                                        format!("invalid value for `{}`: {}", #name_str, e)
                                    ))?
                            ),
                            _ => ::serde_json::from_value(::serde_json::json!(#default_json))
                                .map_err(|e| ::heartbit::Error::Agent(
                                    format!("invalid default for `{}`: {}", #name_str, e)
                                ))?,
                        };
                    }
                } else {
                    quote! {
                        let #name: #ty = match input.get(#name_str) {
                            Some(v) if !v.is_null() => Some(
                                ::serde_json::from_value(v.clone())
                                    .map_err(|e| ::heartbit::Error::Agent(
                                        format!("invalid value for `{}`: {}", #name_str, e)
                                    ))?
                            ),
                            _ => None,
                        };
                    }
                }
            } else {
                quote! {
                    let #name: #ty = {
                        let v = input.get(#name_str).ok_or_else(|| {
                            ::heartbit::Error::Agent(
                                format!("missing required field `{}`", #name_str)
                            )
                        })?;
                        ::serde_json::from_value(v.clone())
                            .map_err(|e| ::heartbit::Error::Agent(
                                format!("invalid value for `{}`: {}", #name_str, e)
                            ))?
                    };
                }
            }
        })
        .collect();

    quote! {
        #(#extractions)*
    }
}

fn lit_to_json_token(lit: &syn::Lit) -> TokenStream2 {
    match lit {
        syn::Lit::Int(i) => {
            let val = i.base10_parse::<i64>().expect("invalid integer literal");
            quote! { #val }
        }
        syn::Lit::Float(f) => {
            let val = f.base10_parse::<f64>().expect("invalid float literal");
            quote! { #val }
        }
        syn::Lit::Str(s) => {
            let val = s.value();
            quote! { #val }
        }
        syn::Lit::Bool(b) => {
            let val = b.value;
            quote! { #val }
        }
        _ => quote! { null },
    }
}
